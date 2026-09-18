"""Render a colour-preserving SVG of the engine's didactic walkthrough.

GitHub Markdown strips ANSI colour from code blocks and sanitises inline HTML
styles, so the only way to show the coloured console output in the README is as
an image. This script runs the walkthrough, captures its ANSI output, and uses
``rich`` to export a terminal-style **SVG** (vector, exact truecolor, crisp at
any zoom) to ``docs/demo.svg``.

``rich`` is a dev-only dependency (not in requirements.txt). Regenerate with::

    pip install rich
    python docs/make_demo_svg.py
"""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.svg"


def capture() -> str:
    """Run the walkthrough (fixed seed) and return its coloured stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "test.test_engine", "--seed", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def excerpt(output: str) -> str:
    """Keep a representative, self-contained slice from the top through STEP 2.

    Starts at the very beginning (config banner, input vector, weights, forward
    pass) and stops just before STEP 3, so the image shows the setup *and* the
    first backprop snapshots up to the matmul step. Selecting whole lines is
    safe because every coloured token opens and closes its ANSI codes on the
    same line.
    """
    lines = output.splitlines()
    end = next(i for i, l in enumerate(lines) if "STEP 3" in l)
    return "\n".join(lines[:end]).rstrip()


def main() -> None:
    console = Console(record=True, width=84)
    console.print(Text.from_ansi(excerpt(capture())))
    console.save_svg(str(OUT), title="bert-cpu — gradient-graph walkthrough")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
