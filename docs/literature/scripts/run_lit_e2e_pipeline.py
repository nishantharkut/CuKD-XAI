"""Download 30-40 literature PDFs, rasterize pages to images, extract page text, write review skeleton.

Usage:
  python docs/literature/scripts/run_lit_e2e_pipeline.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    import fitz  # pymupdf
except ImportError as e:
    raise SystemExit("pip install pymupdf first") from e

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/literature/papers/e2e_manifest.json"
PDF_DIR = ROOT / "docs/literature/papers/e2e_pdfs"
IMG_DIR = ROOT / "docs/literature/papers/e2e_pages"
TEXT_DIR = ROOT / "docs/literature/papers/e2e_text"
LOCAL_PAPERS = ROOT / "docs/literature/papers"
STATUS_PATH = ROOT / "docs/literature/papers/e2e_download_status.json"
REVIEW_PATH = ROOT / "docs/literature/E2E_LITERATURE_REVIEW.md"

HEADERS = {
    "User-Agent": "CuKD-XAI-literature-review/1.0 (research; academic use)",
    "Accept": "application/pdf,*/*",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def download(url: str, dest: Path, timeout: int = 90) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, headers=HEADERS, timeout=timeout, stream=True, allow_redirects=True) as r:
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            if not is_pdf(tmp):
                # sometimes HTML paywall
                head = tmp.read_bytes()[:200]
                tmp.unlink(missing_ok=True)
                return False, f"not PDF (head={head[:80]!r})"
            tmp.replace(dest)
            return True, "ok"
    except Exception as e:
        return False, str(e)


def rasterize(pdf_path: Path, out_dir: Path, dpi: int = 120, max_pages: int = 40) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    n = min(len(doc), max_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    text_chunks = []
    for i in range(n):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"page_{i+1:03d}.png"
        pix.save(str(img_path))
        txt = page.get_text("text")
        text_chunks.append(f"\n\n===== PAGE {i+1} =====\n{txt}")
        pages.append(
            {
                "page": i + 1,
                "image": str(img_path.relative_to(ROOT)).replace("\\", "/"),
                "width": pix.width,
                "height": pix.height,
                "chars": len(txt),
            }
        )
    doc.close()
    return {"n_pages_total": n, "pages": pages, "text": "".join(text_chunks)}


def summarize_text(text: str, max_chars: int = 2500) -> dict:
    # crude extract of abstract-ish and figure captions
    low = text.lower()
    abstract = ""
    m = re.search(r"abstract\s*(.+?)(?:\n\s*1[\.\s]|introduction|keywords)", text, re.I | re.S)
    if m:
        abstract = re.sub(r"\s+", " ", m.group(1)).strip()[:1200]
    figs = re.findall(r"(Figure\s+\d+[\.:].{0,200})", text, re.I)
    tables = re.findall(r"(Table\s+\d+[\.:].{0,200})", text, re.I)
    metrics = re.findall(
        r"((?:accuracy|macro[\-\s]?f1|f1[\-\s]?score|precision|recall|parameters?|kb|flops?|latency|energy)[^\n]{0,120})",
        text,
        re.I,
    )
    return {
        "abstract_excerpt": abstract,
        "figure_captions_sample": figs[:12],
        "table_captions_sample": tables[:12],
        "metric_lines_sample": list(dict.fromkeys(metrics))[:25],
        "text_chars": len(text),
    }


def main() -> int:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    papers = manifest["papers"]
    status = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "downloaded": [],
        "failed": [],
        "local_copied": [],
        "rasterized": [],
        "skipped_dup": [],
    }

    # 1) copy locals first
    local_map = {
        "xiao2025local": "Metaheuristically optimized deep soft-voting ensemble for explainable and resource-aware signal processing in wireless sensor network intrusion detection.pdf",
        "talukder2025local": "sota_wsn_ds_2025.pdf",
        "benaddi2025local": "benaddi_2025.pdf",
        "alfarra2025local": "alfarra_2025.pdf",
    }
    for pid, fname in local_map.items():
        src = LOCAL_PAPERS / fname
        dst = PDF_DIR / f"{pid}.pdf"
        if src.is_file():
            shutil.copy2(src, dst)
            status["local_copied"].append({"id": pid, "src": fname, "bytes": dst.stat().st_size})

    # 2) download remote
    for p in papers:
        pid = p["id"]
        url = p["url"]
        dest = PDF_DIR / f"{pid}.pdf"
        if url.startswith("LOCAL:"):
            continue
        if dest.is_file() and is_pdf(dest):
            status["skipped_dup"].append(pid)
            continue
        print(f"DOWNLOAD {pid} ...")
        ok, msg = download(url, dest)
        if ok:
            status["downloaded"].append(
                {"id": pid, "url": url, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
            )
            print(f"  OK {dest.stat().st_size} bytes")
        else:
            status["failed"].append({"id": pid, "url": url, "error": msg})
            print(f"  FAIL {msg}")
            if dest.exists():
                dest.unlink(missing_ok=True)

    # 3) rasterize every available pdf
    reviews = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"\nRasterizing {len(pdfs)} PDFs ...")
    for pdf in pdfs:
        pid = pdf.stem
        out_img = IMG_DIR / pid
        try:
            info = rasterize(pdf, out_img, dpi=110, max_pages=35)
            text_path = TEXT_DIR / f"{pid}.txt"
            text_path.write_text(info["text"], encoding="utf-8", errors="replace")
            summary = summarize_text(info["text"])
            rec = {
                "id": pid,
                "pdf": str(pdf.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(pdf),
                "bytes": pdf.stat().st_size,
                "n_pages_rasterized": info["n_pages_total"],
                "images_dir": str(out_img.relative_to(ROOT)).replace("\\", "/"),
                "text_path": str(text_path.relative_to(ROOT)).replace("\\", "/"),
                "summary": summary,
            }
            status["rasterized"].append(rec)
            reviews.append(rec)
            print(f"  {pid}: {info['n_pages_total']} pages")
        except Exception as e:
            status["failed"].append({"id": pid, "url": str(pdf), "error": f"rasterize: {e}"})
            print(f"  RASTER FAIL {pid}: {e}")

    status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    status["n_pdfs"] = len(pdfs)
    status["n_downloaded_ok"] = len(status["downloaded"]) + len(status["local_copied"])
    status["n_rasterized"] = len(status["rasterized"])
    status["n_failed"] = len(status["failed"])
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    # 4) write literature review draft from extracted text
    lines = [
        "# E2E Literature Review (page-rasterized corpus)",
        "",
        f"**Generated:** {status['finished']}",
        f"**PDFs available:** {status['n_pdfs']}",
        f"**Rasterized:** {status['n_rasterized']}",
        f"**Failed downloads:** {status['n_failed']}",
        "",
        "Pipeline: download PDF → rasterize pages to PNG → extract text per page → structured notes.",
        "Visual page images live under `docs/literature/papers/e2e_pages/<paper_id>/page_XXX.png`.",
        "",
        "## Corpus inventory",
        "",
        "| id | pages | PDF KB | abstract excerpt |",
        "|---|---:|---:|---|",
    ]
    for r in reviews:
        ab = (r["summary"].get("abstract_excerpt") or "").replace("|", "/")[:180]
        lines.append(
            f"| {r['id']} | {r['n_pages_rasterized']} | {r['bytes']//1024} | {ab} |"
        )
    lines += [
        "",
        "## Per-paper notes (auto-extracted; refine after visual pass)",
        "",
    ]
    for r in reviews:
        s = r["summary"]
        lines.append(f"### {r['id']}")
        lines.append("")
        lines.append(f"- PDF: `{r['pdf']}`")
        lines.append(f"- Pages rasterized: {r['n_pages_rasterized']}")
        lines.append(f"- Images: `{r['images_dir']}/`")
        if s.get("abstract_excerpt"):
            lines.append(f"- Abstract excerpt: {s['abstract_excerpt'][:700]}")
        if s.get("figure_captions_sample"):
            lines.append("- Figure captions (sample):")
            for c in s["figure_captions_sample"][:6]:
                lines.append(f"  - {c.strip()[:200]}")
        if s.get("table_captions_sample"):
            lines.append("- Table captions (sample):")
            for c in s["table_captions_sample"][:6]:
                lines.append(f"  - {c.strip()[:200]}")
        if s.get("metric_lines_sample"):
            lines.append("- Metric-ish lines (sample):")
            for c in s["metric_lines_sample"][:10]:
                lines.append(f"  - {c.strip()[:160]}")
        lines.append("")

    lines += [
        "## Failed / need manual download",
        "",
    ]
    fails = [f for f in status["failed"] if "rasterize" not in f.get("error", "")]
    if not fails:
        lines.append("- (none listed)")
    else:
        for f in fails:
            lines.append(f"- **{f['id']}**: {f.get('error')} — `{f.get('url')}`")
    lines += [
        "",
        "## Positioning for CuKD-XAI (draft)",
        "",
        "1. Accuracy-only WSN/Edge papers often report 95–99% Acc without KB-scale MCU fixed-point HIL.",
        "2. KD compression papers (Benaddi, Yang, LENS-XAI) shrink models; few report dual-board full-test integer agreement.",
        "3. Hardware-aware works (Diab, Alfarra, Javed) target larger flash/SBC or energy; CuKD targets KB dense students + MCU replay.",
        "4. XAI-on-KD is often predictive fidelity or feature pruning, not TreeExplainer–DeepExplainer global rank Spearman on RF-KD deploy units.",
        "5. CuKD contribution after corrections: absolute compression holds; KD benefit is protocol-sensitive; deployment dual-identity + HIL fidelity.",
        "",
    ]
    REVIEW_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSTATUS", json.dumps({k: status[k] for k in ("n_pdfs", "n_downloaded_ok", "n_rasterized", "n_failed")}, indent=2))
    print("Wrote", STATUS_PATH)
    print("Wrote", REVIEW_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
