"""Write review cards from pass1 evidence ONLY. No invented numbers."""
import json, re
from pathlib import Path
from datetime import date

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
EV = ROOT / "_pass1_evidence"
REV = ROOT / "reviews"
REV.mkdir(exist_ok=True)
EX = ROOT / "_extract"

# freeze tags for positioning hints (not claims about paper content)
def cukd_tags(pid, abstract, metrics):
    blob = (abstract + " " + " ".join(metrics)).lower()
    tags = []
    if any(k in blob for k in ["distill", "knowledge distillation", "teacher", "student"]):
        tags.append("KD")
    if any(k in blob for k in ["shap", "explain", "xai", "interpret"]):
        tags.append("XAI")
    if any(k in blob for k in ["esp32", "arduino", "mcu", "microcontroller", "tinyml", "embedded"]):
        tags.append("MCU/embedded")
    if any(k in blob for k in ["wsn-ds", "wsn ds", "blackhole", "grayhole", "wireless sensor"]):
        tags.append("WSN")
    if any(k in blob for k in ["edge-iiot", "edgeiiot", "iiot"]):
        tags.append("Edge-IIoT")
    if any(k in blob for k in ["quantiz", "integer", "fixed-point", "int8"]):
        tags.append("quantization")
    if any(k in blob for k in ["federat"]):
        tags.append("federated")
    if any(k in blob for k in ["raspberry", "gateway"]):
        tags.append("gateway/SBC")
    return tags

def pick_title(title_lines):
    # longest reasonable title-like line among first few
    cands = []
    for ln in title_lines[:15]:
        if 20 <= len(ln) <= 220 and not re.match(r"^[A-Z]\.\s", ln):
            # skip pure author lines with emails
            if "@" in ln:
                continue
            if re.search(r"\d{4}", ln) and len(ln) < 40:
                continue
            cands.append(ln)
    if not cands:
        return title_lines[0] if title_lines else "TITLE_NOT_EXTRACTED"
    # prefer longest early candidate
    return max(cands[:8], key=len)

def write_card(ev):
    pid = ev["id"]
    title = pick_title(ev.get("title_lines") or [])
    abs_ = ev.get("abstract") or ""
    tables = ev.get("table_snippets") or []
    metrics = ev.get("metric_lines") or []
    tags = cukd_tags(pid, abs_, metrics)
    pages = ev.get("n_pages")
    res_pages = ev.get("resultish_pages") or []

    # authors: lines after title that look like names
    authors = []
    for ln in (ev.get("title_lines") or [])[1:12]:
        if ln == title:
            continue
        if re.search(r"(university|department|college|institute|school of|china|usa|uk|email)", ln, re.I):
            break
        if 5 < len(ln) < 120 and not re.search(r"\d{5}", ln):
            authors.append(ln)
        if len(authors) >= 3:
            break

    lines = []
    lines.append(f"# Visual/PDF review card: {pid}")
    lines.append("")
    lines.append(f"**Status:** PASS1_DONE (text extract + structured evidence; PASS2 pending)")
    lines.append(f"**PDF pages:** {pages}")
    lines.append(f"**Result-ish pages (heuristic):** {res_pages[:15]}")
    lines.append(f"**Evidence source:** `_pass1_evidence/{pid}.json` + `_extract/{pid}.full.txt`")
    lines.append(f"**Date:** {date.today().isoformat()}")
    lines.append("")
    lines.append("## Identity (from page-1 text extract)")
    lines.append(f"- **Title (extracted):** {title}")
    if authors:
        lines.append(f"- **Author lines (extracted):** {'; '.join(authors[:3])}")
    lines.append(f"- **Theme tags (heuristic):** {', '.join(tags) if tags else 'general'}")
    lines.append("")
    lines.append("## Abstract (verbatim extract; truncated)")
    if abs_:
        lines.append("")
        lines.append(f"> {abs_[:1500]}")
    else:
        lines.append("")
        lines.append("_Abstract block not auto-detected; see full extract._")
    lines.append("")
    lines.append("## Table snippets (verbatim from PDF text)")
    if tables:
        for i, t in enumerate(tables[:6], 1):
            snip = t.replace("\n", " / ")
            lines.append(f"{i}. `{snip[:500]}`")
    else:
        lines.append("_No 'Table N' headers found in text layer (may be image-only tables)._")
    lines.append("")
    lines.append("## Metric-bearing lines (verbatim samples)")
    if metrics:
        for m in metrics[:25]:
            lines.append(f"- `{m}`")
    else:
        lines.append("_No metric-like lines auto-collected._")
    lines.append("")
    lines.append("## CuKD positioning notes (non-numeric; freeze-aware)")
    notes = []
    if "KD" in tags:
        notes.append("Related to knowledge distillation compression (C1/C2 literature neighborhood).")
    if "XAI" in tags:
        notes.append("Related to XAI/SHAP narrative; do not invent Spearman results for this paper.")
    if "MCU/embedded" in tags:
        notes.append("On-device/MCU-related; compare carefully to C4 dual-board RF-KD HIL.")
    if "WSN" in tags:
        notes.append("WSN/WSN-DS neighborhood; Almomani dataset lineage may apply.")
    if "Edge-IIoT" in tags:
        notes.append("Edge-IIoT neighborhood; connect to C10 group-aware discussion if used.")
    if "quantization" in tags:
        notes.append("Quantization-related; anchor to Jacob/C4 PTQ honesty.")
    if "federated" in tags:
        notes.append("Federated learning setting; distinct from single-node MCU HIL.")
    if not notes:
        notes.append("Foundational or survey/background cite; use only for context, not as CuKD SOTA rival unless method overlaps.")
    notes.append("**Forbidden:** claiming numbers not present above; claiming visual figures without image pass.")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## PASS2 checklist")
    lines.append("- [ ] Title confirmed against page_001.png")
    lines.append("- [ ] Every quoted metric re-found in `_extract/{id}.full.txt`")
    lines.append("- [ ] Method/results interpretation checked on ≥1 results page image if tables image-only")
    lines.append("")
    text = "\n".join(lines)
    (REV / f"{pid}.md").write_text(text, encoding="utf-8")
    return pid

ids = sorted(p.stem for p in EV.glob("*.json") if p.name != "_summary.json")
done = []
for pid in ids:
    ev = json.loads((EV / f"{pid}.json").read_text(encoding="utf-8"))
    write_card(ev)
    done.append(pid)
    print("CARD", pid)

print("WROTE", len(done), "cards")
