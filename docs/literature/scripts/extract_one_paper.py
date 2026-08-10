"""One-paper evidence extraction: full text dump + page index. No LLM claims."""
import fitz, json, re, sys
from pathlib import Path

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
PDF_DIR = ROOT / "e2e_pdfs"
OUT_DIR = ROOT / "_extract"
OUT_DIR.mkdir(exist_ok=True)

def extract_one(pid: str):
    pdf = PDF_DIR / f"{pid}.pdf"
    if not pdf.exists():
        return {"id": pid, "error": "missing pdf"}
    doc = fitz.open(pdf)
    pages = []
    full = []
    for i in range(len(doc)):
        t = doc[i].get_text("text")
        pages.append({"page": i+1, "n_chars": len(t), "text": t})
        full.append(f"\n\n===== PAGE {i+1} =====\n\n{t}")
    meta = doc.metadata or {}
    doc.close()
    all_text = "\n".join(full)
    # crude title: first non-empty lines of page 1
    p1 = pages[0]["text"] if pages else ""
    lines = [ln.strip() for ln in p1.splitlines() if ln.strip()]
    # find abstract block
    abs_m = re.search(r"(?is)\babstract\b[:\s\-]*\n?(.*?)(?:\n\s*(?:1[\.\s]+introduction|keywords|index terms|1 introduction))", p1)
    abstract = abs_m.group(1).strip()[:2500] if abs_m else ""
    # pages mentioning table/result keywords
    hit_pages = []
    for pg in pages:
        low = pg["text"].lower()
        keys = ["table", "accuracy", "f1", "macro", "precision", "recall", "parameter", "latency", "esp32", "flash", "deploy", "distill", "shap"]
        score = sum(1 for k in keys if k in low)
        if score >= 2:
            hit_pages.append(pg["page"])
    rec = {
        "id": pid,
        "n_pages": len(pages),
        "bytes": pdf.stat().st_size,
        "metadata": meta,
        "page1_first_lines": lines[:40],
        "abstract_guess": abstract,
        "resultish_pages": hit_pages[:25],
        "pages": [{"page": p["page"], "n_chars": p["n_chars"]} for p in pages],
    }
    # write full text for pass2
    (OUT_DIR / f"{pid}.full.txt").write_text(all_text, encoding="utf-8", errors="replace")
    (OUT_DIR / f"{pid}.meta.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    # also per-page text files for targeted reads
    pdir = OUT_DIR / pid
    pdir.mkdir(exist_ok=True)
    for p in pages:
        (pdir / f"page_{p['page']:03d}.txt").write_text(p["text"], encoding="utf-8", errors="replace")
    return rec

if __name__ == "__main__":
    # process one or all
    if len(sys.argv) > 1:
        ids = sys.argv[1:]
    else:
        ids = [p.stem for p in sorted(PDF_DIR.glob("*.pdf"))]
        # skip dups
        skip = {"benaddi2025local", "talukder2025local"}
        ids = [i for i in ids if i not in skip]
    summary = []
    for pid in ids:
        try:
            r = extract_one(pid)
            summary.append({"id": pid, "n_pages": r.get("n_pages"), "ok": "error" not in r})
            print(f"OK {pid} pages={r.get('n_pages')} hits={r.get('resultish_pages', [])[:8]}")
        except Exception as e:
            summary.append({"id": pid, "ok": False, "error": str(e)})
            print(f"FAIL {pid}: {e}")
    (OUT_DIR / "_extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", sum(1 for s in summary if s.get("ok")), "/", len(summary))
