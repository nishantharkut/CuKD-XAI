import json, re
from pathlib import Path

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
EX = ROOT / "_extract"
OUT = ROOT / "_pass1_evidence"
OUT.mkdir(exist_ok=True)

def first_title_block(p1: str, n=25):
    lines = [ln.strip() for ln in p1.splitlines() if ln.strip()]
    # drop common headers
    skip_re = re.compile(r"^(sensors|computers|electronics|future internet|ieee|mdpi|article|research article|original article|open access|https?://|doi:|www\.|volume |received:|accepted:|published:|citation:|academic editor|check for updates|licensee|copyright|this article|keywords:|index terms)", re.I)
    keep = []
    for ln in lines[:80]:
        if skip_re.match(ln):
            continue
        if len(ln) < 3:
            continue
        keep.append(ln)
        if len(keep) >= n:
            break
    return keep

def abstract_block(text: str):
    m = re.search(r"(?is)\babstract\b\s*[:\—\-]?\s*(.*?)(?:\n\s*(?:keywords|index terms|1[\.\s]+introduction|i\.\s+introduction|1 introduction))", text[:8000])
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:2000]
    return ""

def pull_tables(full: str, max_tables=8):
    # grab chunks around "Table N"
    hits = []
    for m in re.finditer(r"(?im)^(.*\btable\s+[0-9ivx]+.*)$", full):
        start = m.start()
        chunk = full[start:start+1200]
        hits.append(re.sub(r"[ \t]+", " ", chunk)[:900])
        if len(hits) >= max_tables:
            break
    return hits

def metric_lines(full: str, limit=40):
    lines = []
    for ln in full.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.search(r"(?i)(accuracy|macro.?f1|f1.?score|precision|recall|parameter|latency|esp32|flash|rho|spearman|distill)", s) and re.search(r"\d", s):
            if len(s) < 220:
                lines.append(s)
        if len(lines) >= limit:
            break
    return lines

skip = {"benaddi2025local", "talukder2025local"}
ids = sorted([p.stem for p in (ROOT/"e2e_pdfs").glob("*.pdf") if p.stem not in skip])

summary = []
for pid in ids:
    meta_p = EX / f"{pid}.meta.json"
    full_p = EX / f"{pid}.full.txt"
    if not full_p.exists():
        summary.append({"id": pid, "ok": False, "error": "no extract"})
        continue
    full = full_p.read_text(encoding="utf-8", errors="replace")
    # page 1 only for title/abstract
    p1m = re.search(r"===== PAGE 1 =====\s*(.*?)(?:===== PAGE 2 =====|$)", full, re.S)
    p1 = p1m.group(1) if p1m else full[:5000]
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    ev = {
        "id": pid,
        "n_pages": meta.get("n_pages"),
        "title_lines": first_title_block(p1),
        "abstract": abstract_block(p1 if "abstract" in p1.lower() else full[:10000]),
        "resultish_pages": meta.get("resultish_pages", []),
        "table_snippets": pull_tables(full),
        "metric_lines": metric_lines(full),
        "pass": 1,
        "source": "pymupdf_text_extract",
    }
    (OUT / f"{pid}.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    summary.append({"id": pid, "ok": True, "n_pages": ev["n_pages"], "n_tables": len(ev["table_snippets"]), "n_metrics": len(ev["metric_lines"])})
    print(f"P1 {pid}: pages={ev['n_pages']} tables={len(ev['table_snippets'])} metrics={len(ev['metric_lines'])}")

(OUT/"_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("PASS1 evidence JSON:", sum(1 for s in summary if s["ok"]), "/", len(summary))
