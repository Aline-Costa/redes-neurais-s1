"""Shared console styling for the didactic walkthroughs.

The ``learn`` walkthroughs use a single colour scheme, organised by *content
category* (not by colour) so that restyling a category is a one-line change.
Everything is encapsulated in the :class:`Console` class and exposed as the
module-level singleton :data:`console`, which any walkthrough can import::

    from learn.console import console

    print(console.label("name"), console.fmt(some_array))

Categories
----------
``text``   prose / explanatory sentences
``math``   variables and equations (x, w, y = tanh(w . x), dy/dz, ...)
``label``  reserved keywords (node name, value, grad, STEP)
``value``  numeric values and matrices (the actual numbers)
"""

from __future__ import annotations

import os

import numpy as np


class Console:
    """Content-category colour scheme plus small formatting helpers.

    To restyle a category, change only its ANSI SGR code in :attr:`COLOR`
    (https://en.wikipedia.org/wiki/ANSI_escape_code#SGR). An empty code leaves
    the text in the terminal's default colour. ``NO_COLOR`` disables all colour.
    """

    #: ANSI SGR code per content category (the single source of truth).
    COLOR = {
        "text": "",                       # default / white
        "math": "1;31",                   # bold red
        "label": "38;2;217;119;87",       # Anthropic orange (#D97757, truecolor)
        "value": "1;38;2;189;147;249",    # bold Dracula violet (#BD93F9, truecolor)
    }

    # ------------------------------------------------------------------ #
    # Colouring
    # ------------------------------------------------------------------ #
    def paint(self, category: str, text: str) -> str:
        """Wrap ``text`` in the ANSI code registered for ``category``.

        No-op when the category has no code or when ``NO_COLOR`` is set, so the
        output stays free of escape codes when redirected to a file.
        """
        code = self.COLOR[category]
        if not code or os.environ.get("NO_COLOR"):
            return text
        return f"\033[{code}m{text}\033[0m"

    def text(self, s: str) -> str:
        """Colour prose / explanatory text."""
        return self.paint("text", s)

    def math(self, s: str) -> str:
        """Colour maths: variables and equations."""
        return self.paint("math", s)

    def label(self, s: str) -> str:
        """Colour the reserved keywords (``node name``, ``value``, ``grad``, ``STEP``)."""
        return self.paint("label", s)

    def value(self, s: str) -> str:
        """Colour numeric values and matrices."""
        return self.paint("value", s)

    # ------------------------------------------------------------------ #
    # Number formatting
    # ------------------------------------------------------------------ #
    def fmt_plain(self, value: np.ndarray) -> str:
        """Compact, sign-aligned formatting of a scalar or small array (no colour).

        Returning the uncoloured string lets callers measure its real width for
        table alignment before wrapping it in ANSI codes.
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            return f"{float(arr):+.3f}"
        return "[" + ", ".join(f"{v:+.3f}" for v in arr.ravel()) + "]"

    def fmt(self, value: np.ndarray) -> str:
        """Like :meth:`fmt_plain`, but coloured as the ``value`` category."""
        return self.value(self.fmt_plain(value))

    def fmt_matrix_plain(self, value: np.ndarray, indent: str = "") -> str:
        """Format a 2D array as aligned, bracketed rows (a maths-style matrix).

        Each row becomes ``[  a   b ]`` with the columns right-aligned to a common
        width, and rows stacked vertically — so a weight matrix reads like the
        matrix it is, instead of a single flattened ``ravel()`` list. A scalar or
        1D array is laid out as a single column (the column-vector convention used
        throughout the demos). Uncoloured, so callers can measure or recolour it.
        """
        arr = np.asarray(value)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)          # a vector is a single column
        cells = [[f"{v:+.3f}" for v in row] for row in arr]
        col_w = [max(len(cells[r][c]) for r in range(len(cells))) for c in range(arr.shape[1])]
        lines = [
            indent + "[ " + "   ".join(c.rjust(col_w[i]) for i, c in enumerate(row)) + " ]"
            for row in cells
        ]
        return "\n".join(lines)

    def fmt_matrix(self, value: np.ndarray, indent: str = "") -> str:
        """Like :meth:`fmt_matrix_plain`, but coloured as the ``value`` category."""
        return self.value(self.fmt_matrix_plain(value, indent))

    def hjoin(self, blocks, gap: str = " ") -> str:
        """Lay several text blocks side by side, each vertically centred.

        Each block is a ``(text, colour)`` pair: ``text`` may be multi-line (e.g.
        a matrix from :meth:`fmt_matrix_plain`) and ``colour`` is a colour method
        (or ``None``). Shorter blocks are padded top and bottom so all blocks
        share the same height — which is what lets us write ``A @ B = C`` with the
        operators sitting on the middle row between the matrices.
        """
        grids = [(text.split("\n"), colour) for text, colour in blocks]
        widths = [max((len(ln) for ln in lines), default=0) for lines, _ in grids]
        height = max((len(lines) for lines, _ in grids), default=1)
        columns = []
        for (lines, colour), w in zip(grids, widths):
            pad = height - len(lines)
            top = pad // 2
            padded = [""] * top + lines + [""] * (pad - top)
            painted = [
                colour(ln.ljust(w)) if (colour and ln.strip()) else ln.ljust(w)
                for ln in padded
            ]
            columns.append(painted)
        return "\n".join(gap.join(col[r] for col in columns) for r in range(height))

    # ------------------------------------------------------------------ #
    # Computational-graph drawing (shared by all walkthroughs)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _graph_name(node) -> str:
        """A node's display name: its ``label`` if set, else its op, else ``leaf``."""
        return getattr(node, "label", None) or getattr(node, "_op", "") or "leaf"

    @staticmethod
    def _graph_children(node) -> list:
        """A node's operands (the tensors it was built from), in a stable order.

        The engine stores parents in an unordered ``set`` (``_prev``); we sort by
        name then ``id`` so the drawing is deterministic from run to run.
        """
        prev = getattr(node, "_prev", None)
        if not prev:
            return []
        return sorted(
            prev,
            key=lambda c: (getattr(c, "label", "") or getattr(c, "_op", "") or "", id(c)),
        )

    def fmt_auto_plain(self, value: np.ndarray, indent: str = "") -> str:
        """Format any non-scalar as a matrix table, the way maths writes it.

        Every vector/matrix is laid out with :meth:`fmt_matrix_plain`, so it
        matches its mathematical shape: a **column vector** ``(n, 1)`` stacks
        vertically, a **row vector** ``(1, n)`` stays on one line, and a general
        matrix ``(n, m)`` is a grid. Only a single scalar stays inline as
        ``+x.xxx`` (:meth:`fmt_plain`).
        """
        arr = np.asarray(value)
        if arr.size <= 1:
            return indent + self.fmt_plain(arr)
        if arr.ndim <= 2:
            return self.fmt_matrix_plain(arr, indent)
        return indent + self.fmt_plain(arr)   # 3D+ fallback: flat list

    def fmt_auto(self, value: np.ndarray, indent: str = "") -> str:
        """Like :meth:`fmt_auto_plain`, but coloured as the ``value`` category."""
        return self.value(self.fmt_auto_plain(value, indent))

    def format_table(self, rows, headers=None, col_colors=None,
                     leads=None, lead_conts=None, notes=None) -> str:
        """Render aligned columns where any cell may span multiple lines.

        ``rows`` is a list of rows, each a list of cell strings. A cell may
        itself contain newlines (e.g. a matrix laid out by
        :meth:`fmt_matrix_plain` / :meth:`fmt_auto_plain`); columns are sized to
        their widest line and multi-line cells push the row down by extra lines,
        with the other columns left blank on those lines. A ``"-"`` cell is left
        uncoloured (used for "not computed yet").

        ``leads`` / ``lead_conts`` add an optional left-hand gutter per row (used
        by :meth:`format_graph` for the tree connectors): ``leads[r]`` prints on
        the row's first line and ``lead_conts[r]`` on its continuation lines.
        ``notes[r]`` is appended (uncoloured) after the first line.
        """
        n = len(rows)
        ncol = max([len(r) for r in rows] + ([len(headers)] if headers else []), default=0)

        # Split every cell into its physical lines, for measuring and rendering.
        def cell_lines(r, c):
            return rows[r][c].split("\n") if c < len(rows[r]) else [""]

        split = [[cell_lines(r, c) for c in range(ncol)] for r in range(n)]

        col_w = []
        for c in range(ncol):
            w = max((len(ln) for r in range(n) for ln in split[r][c]), default=0)
            if headers and c < len(headers):
                w = max(w, len(headers[c]))
            col_w.append(w)

        has_lead = leads is not None
        lead_w = 0
        if has_lead:
            lead_w = max([len(x) for x in leads] +
                         [len(x) for x in (lead_conts or [])] +
                         ([len("graph")] if headers else [0]))

        col_colors = col_colors or ([self.math] + [self.value] * (ncol - 1))

        def render_line(line_idx, row_split):
            cells = []
            for c in range(ncol):
                seg = row_split[c][line_idx] if line_idx < len(row_split[c]) else ""
                padded = seg.ljust(col_w[c])
                colour = col_colors[c] if c < len(col_colors) else self.text
                cells.append(padded if (not seg or seg == "-") else colour(padded))
            return "  ".join(cells)

        out: list = []
        if headers:
            head = (self.label("graph".ljust(lead_w)) + "  ") if has_lead else ""
            head += "  ".join(
                self.label((headers[c] if c < len(headers) else "").ljust(col_w[c]))
                for c in range(ncol)
            )
            out.append(head)
            sep = ("-" * lead_w + "--" if has_lead else "") + \
                  "--".join("-" * col_w[c] for c in range(ncol))
            out.append(sep)

        for r in range(n):
            height = max((len(split[r][c]) for c in range(ncol)), default=1)
            for li in range(height):
                line = ""
                if has_lead:
                    gutter = leads[r] if li == 0 else (lead_conts[r] if lead_conts else "")
                    line += self.text(gutter.ljust(lead_w)) + "  "
                line += render_line(li, split[r])
                if notes and li == 0 and notes[r]:
                    line += self.text("   " + notes[r])
                out.append(line.rstrip())
        return "\n".join(out)

    def format_graph(self, root, columns=None, headers=None, col_colors=None) -> str:
        """Draw the computational graph that produced ``root`` as a vertical tree.

        Reads the autograd graph straight from a tensor — ``root`` is the output,
        printed at the top, and each operand (from ``_prev``) hangs **below** its
        consumer with box-drawing connectors, the way ``git log --graph`` stacks
        commits above their parents::

            ● y = tanh        value=[...]   grad=[...]
            └─ ● z = @         value=[...]   grad=[...]
               ├─ ● wᵀ         value=[...]   grad=[...]
               │  └─ ● w        value=[...]   grad=[...]
               └─ ● x           value=[...]   grad=[...]

        so a reader can follow, line by line, which tensors flow into which op.

        The graph is a DAG: a tensor used in several places would be expanded once
        (at its first appearance) and then shown as a back-reference, to avoid
        redrawing a whole shared sub-graph.

        Parameters
        ----------
        root : Tensor
            The output node; the rest of the graph is discovered through ``_prev``.
        columns : callable, optional
            ``node -> list[str]`` returning the cells to show beside each node
            (e.g. ``[name, value, grad]``). Cells are aligned into columns across
            all rows. Defaults to a single ``name`` column.
        headers : list[str], optional
            Column titles printed above the tree (aligned with ``columns``).
        col_colors : list[callable], optional
            One colour function per column (defaults to ``math`` for the first
            column and ``value`` for the rest). A ``"-"`` cell is left uncoloured.

        Returns
        -------
        str
            The full, coloured, multi-line drawing.
        """
        columns = columns or (lambda n: [self._graph_name(n)])

        # Walk the graph depth-first, building the per-row cells plus the tree
        # gutter (``leads`` on a node's first line, ``lead_conts`` on the extra
        # lines of a multi-line cell). ``format_table`` does the alignment.
        rows: list = []
        leads: list = []
        lead_conts: list = []
        notes: list = []
        seen: set = set()

        def walk(node, prefix: str, is_last: bool, is_root: bool) -> None:
            if is_root:
                graph_part = "● "
                child_prefix = ""
            else:
                graph_part = prefix + ("└─ " if is_last else "├─ ") + "● "
                child_prefix = prefix + ("   " if is_last else "│  ")

            kids = self._graph_children(node)
            ref = id(node) in seen and bool(kids)
            rows.append(columns(node))
            leads.append(graph_part)
            lead_conts.append(child_prefix)
            notes.append("⤴ (shown above)" if ref else "")
            if ref:
                # Already drawn elsewhere: back-reference, do not recurse.
                return
            seen.add(id(node))
            for i, kid in enumerate(kids):
                walk(kid, child_prefix, i == len(kids) - 1, False)

        walk(root, "", True, True)
        return self.format_table(rows, headers=headers, col_colors=col_colors,
                                 leads=leads, lead_conts=lead_conts, notes=notes)

    def print_graph(self, root, **kwargs) -> None:
        """Print :meth:`format_graph` (convenience wrapper)."""
        print(self.format_graph(root, **kwargs))

    # ------------------------------------------------------------------ #
    # Printing
    # ------------------------------------------------------------------ #
    def kv(self, label: str, value: str, note: str = "", color=None) -> None:
        """Print one ``label: value`` line in the colour scheme.

        ``label`` is printed with ``color`` (the ``text`` category by default,
        or :meth:`math` for equations) and any trailing ``note`` with ``text``,
        while ``value`` is expected to already carry its own colour (e.g. from
        :meth:`fmt`).

        If ``value`` spans multiple lines (e.g. a column vector or matrix from
        :meth:`fmt_auto`), it is dropped onto the lines *below* the label — the
        way a vector is written under its name in maths — with the ``note`` kept
        on the label line.
        """
        color = color or self.text
        if "\n" in value:
            head = color(label.rstrip())
            print(head + (self.text("  " + note.strip()) if note else ""))
            # Nest the block (a column vector / matrix) under the label.
            print("\n".join("        " + ln for ln in value.split("\n")))
        else:
            print(color(label), end="")
            print(value, end="")
            print(self.text(note) if note else "")


#: Module-level singleton shared by every test file.
console = Console()
