"""Locate Graphviz `dot` on Windows and expose helpers for professional DOT diagrams."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Common install locations (winget Graphviz.Graphviz)
_CANDIDATES = [
    Path(r"C:\Program Files\Graphviz\bin"),
    Path(r"C:\Program Files (x86)\Graphviz\bin"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Graphviz" / "bin",
]


def ensure_graphviz_on_path() -> Path:
    """Return directory containing dot.exe; prepend to PATH if needed."""
    which = shutil.which("dot")
    if which:
        return Path(which).parent
    for d in _CANDIDATES:
        if d and (d / "dot.exe").is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return d
    raise FileNotFoundError(
        "Graphviz `dot` not found. Install with: winget install Graphviz.Graphviz\n"
        "Then restart the shell or ensure C:\\Program Files\\Graphviz\\bin is on PATH."
    )


def dot_version() -> str:
    ensure_graphviz_on_path()
    out = subprocess.check_output(["dot", "-V"], stderr=subprocess.STDOUT, text=True)
    return out.strip()


def render_dot(
    source: str,
    out_stem: Path,
    formats: tuple[str, ...] = ("pdf", "png", "svg"),
    engine: str = "dot",
) -> list[Path]:
    """Write .dot source and render with Graphviz engine. Returns list of output paths."""
    ensure_graphviz_on_path()
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_stem.with_suffix(".dot")
    dot_path.write_text(source, encoding="utf-8")
    written: list[Path] = [dot_path]
    for fmt in formats:
        out = out_stem.with_suffix(f".{fmt}")
        cmd = [engine, f"-T{fmt}", str(dot_path), "-o", str(out)]
        subprocess.check_call(cmd)
        written.append(out)
    return written
