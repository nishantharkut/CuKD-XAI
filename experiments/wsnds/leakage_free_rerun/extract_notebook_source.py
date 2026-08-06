"""Extract code cells from an archived notebook into a separate runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    header = (
        "# Generated from archived notebook code cells.\n"
        f"# Source: {args.notebook.as_posix()}\n"
        "# This generated copy is intentionally edited only in the train-only-scaler workspace.\n\n"
    )
    body = "\n\n".join(code_cells).replace("\r\n", "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + body + "\n", encoding="utf-8", newline="\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
