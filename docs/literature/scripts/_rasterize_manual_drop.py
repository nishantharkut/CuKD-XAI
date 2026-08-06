from pathlib import Path
import fitz
import json

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI")
PDF_DIR = ROOT / "docs/literature/papers/e2e_pdfs"
IMG_DIR = ROOT / "docs/literature/papers/e2e_pages"
NEW_IDS = [
    "javed2024thermostat","wisanwanichthan2025kd","peng2025fdids","almomani2016wsnds",
    "nguyen2024gswo","alqahtani2019gxgboost","birahim2025pso","ghadi2024review","alshehri2024sadcnn",
    "seyedkolaei2025cnn","gao2026lightweight","adjewa2026seed","ishtiaq2025cstafnet","salmi2022cnnlstm",
    "chawla2002smote","ticnna_hybrid_iot",
]

def rasterize(pdf_path: Path, out_dir: Path, dpi: int = 110, max_pages: int = 35):
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear old pages
    for old in out_dir.glob("page_*.png"):
        try:
            old.unlink()
        except Exception as e:
            print("warn unlink", old, e)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    n = min(len(doc), max_pages)
    for i in range(n):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_dir / f"page_{i+1:03d}.png"))
    total = len(doc)
    doc.close()
    return {"n_pages_rasterized": n, "n_pages_total": total}

results = []
for pid in NEW_IDS:
    pdf = PDF_DIR / f"{pid}.pdf"
    if not pdf.exists():
        print("MISSING", pid)
        continue
    out = IMG_DIR / pid
    try:
        info = rasterize(pdf, out)
        results.append({"id": pid, **info, "bytes": pdf.stat().st_size, "ok": True})
        print(f"OK {pid}: {info['n_pages_rasterized']}/{info['n_pages_total']} pages")
    except Exception as e:
        results.append({"id": pid, "ok": False, "error": str(e)})
        print(f"FAIL {pid}: {e}")

outj = ROOT / "docs/literature/papers/manual_drop_raster_status.json"
outj.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
print("wrote", outj)
print("done", sum(1 for r in results if r.get("ok")), "/", len(results))
