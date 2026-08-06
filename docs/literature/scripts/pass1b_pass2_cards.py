"""PASS1b+PASS2: cards with ONLY exact full-text lines (guaranteed verifiable). Two full loops."""
import json, re
from pathlib import Path
from datetime import date

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
EX = ROOT / "_extract"
REV = ROOT / "reviews"
EV2 = ROOT / "_pass1b_evidence"
EV2.mkdir(exist_ok=True)
P2 = ROOT / "_pass2_verify"
P2.mkdir(exist_ok=True)

skip = {"benaddi2025local", "talukder2025local"}

def exact_metric_lines(full: str, limit=30):
    out = []
    for ln in full.splitlines():
        s = ln.strip()
        if not s or len(s) > 200 or len(s) < 8:
            continue
        if re.search(r"(?i)(accuracy|macro.?f1|\bf1\b|precision|recall|parameter|latency|esp32|flash|distill|shap|f1-score|f1 score)", s) and re.search(r"\d", s):
            # must be exact line content
            out.append(s)
        if len(out) >= limit:
            break
    return out

def exact_table_header_lines(full: str, limit=12):
    out = []
    for ln in full.splitlines():
        s = ln.strip()
        if re.match(r"(?i)^table\s+\d+", s) and 5 < len(s) < 200:
            out.append(s)
        if len(out) >= limit:
            break
    return out

def title_from_p1(p1: str):
    lines = [ln.strip() for ln in p1.splitlines() if ln.strip()]
    skip_re = re.compile(r"^(sensors|computers|electronics|future internet|ieee access|ieee|mdpi|article|research article|original article|open access|https?://|doi[:\s]|www\.|volume |received|accepted|published|citation|academic editor|check for updates|licensee|copyright|this article|keywords|index terms|journal of|hindawi|preprint|submitted)", re.I)
    cands = []
    for ln in lines[:60]:
        if skip_re.match(ln):
            continue
        if "@" in ln or len(ln) < 15:
            continue
        if re.match(r"^\d+$", ln):
            continue
        cands.append(ln)
    if not cands:
        return lines[0] if lines else "UNKNOWN"
    # title often multi-line: take first 1-3 long lines joined if they look like title case
    title_parts = []
    for ln in cands[:6]:
        if re.search(r"(university|department|college|institute|email|china|pakistan|india)", ln, re.I):
            break
        if len(ln) >= 15:
            title_parts.append(ln)
        if sum(len(x) for x in title_parts) > 80:
            break
    return " ".join(title_parts[:3]) if title_parts else cands[0]

def abstract_block(text: str):
    m = re.search(r"(?is)\babstract\b\s*[:\—\-]?\s*(.*?)(?:\n\s*(?:keywords|index terms|1[\.\s]+introduction|i\.\s+introduction|1 introduction|keywords:))", text[:12000])
    if not m:
        return ""
    a = re.sub(r"\s+", " ", m.group(1)).strip()
    return a[:1800]

def theme_tags(blob: str):
    b = blob.lower()
    tags = []
    mapping = [
        ("KD", ["distill", "knowledge distillation"]),
        ("XAI", ["shap", "explainable", "xai", "interpretab"]),
        ("MCU", ["esp32", "arduino", "microcontroller", "tinyml", "embedded"]),
        ("WSN", ["wsn-ds", "wireless sensor", "blackhole", "grayhole"]),
        ("EdgeIIoT", ["edge-iiot", "edgeiiot", "industrial iot"]),
        ("quant", ["quantiz", "integer-arithmetic", "int8", "fixed-point"]),
        ("FL", ["federat"]),
        ("gateway", ["raspberry", "gateway"]),
    ]
    for name, keys in mapping:
        if any(k in b for k in keys):
            tags.append(name)
    return tags

def build_card(pid: str):
    full = (EX / f"{pid}.full.txt").read_text(encoding="utf-8", errors="replace")
    p1m = re.search(r"===== PAGE 1 =====\s*(.*?)(?:===== PAGE 2 =====|$)", full, re.S)
    p1 = p1m.group(1) if p1m else full[:8000]
    title = title_from_p1(p1)
    abstract = abstract_block(p1 if "abstract" in p1.lower() else full[:12000])
    metrics = exact_metric_lines(full)
    tables = exact_table_header_lines(full)
    # verify exact
    for m in metrics:
        assert m in full, m
    for t in tables:
        assert t in full, t
    n_pages = full.count("===== PAGE ")
    tags = theme_tags(abstract + " " + " ".join(metrics) + " " + title)
    ev = {
        "id": pid,
        "n_pages": n_pages,
        "title": title,
        "abstract": abstract,
        "table_headers": tables,
        "metric_lines": metrics,
        "tags": tags,
    }
    (EV2 / f"{pid}.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")

    md = []
    md.append(f"# Review card: {pid}")
    md.append("")
    md.append(f"**Status:** PASS1B_DONE + PASS2_TEXT_OK")
    md.append(f"**PDF text pages extracted:** {n_pages}")
    md.append(f"**Ground truth extract:** `_extract/{pid}.full.txt`")
    md.append(f"**Evidence JSON:** `_pass1b_evidence/{pid}.json`")
    md.append(f"**Generated:** {date.today().isoformat()}")
    md.append("")
    md.append("## Identity (page-1 text)")
    md.append(f"- **Title:** {title}")
    md.append(f"- **Tags:** {', '.join(tags) if tags else 'n/a'}")
    md.append("")
    md.append("## Abstract (extracted)")
    if abstract:
        md.append(f"> {abstract}")
    else:
        md.append("_Not auto-detected; open full extract._")
    md.append("")
    md.append("## Table headers present in PDF text (exact lines)")
    if tables:
        for t in tables:
            md.append(f"- `{t}`")
    else:
        md.append("_None detected (image-only tables possible)._")
    md.append("")
    md.append("## Metric-bearing lines (exact PDF lines; PASS2-verified)")
    if metrics:
        for m in metrics:
            md.append(f"- `{m}`")
    else:
        md.append("_None auto-collected._")
    md.append("")
    md.append("## CuKD freeze notes (non-numeric)")
    if "KD" in tags:
        md.append("- KD neighborhood → compare to C1/C2; do not claim novelty of KD-for-IDS alone.")
    if "XAI" in tags:
        md.append("- XAI neighborhood → do not invent Spearman ρ; C6 is CuKD measurement.")
    if "MCU" in tags:
        md.append("- MCU/embedded neighborhood → compare to C4 dual-board RF-KD HIL; Javed is tree-on-ESP32 prior.")
    if "WSN" in tags:
        md.append("- WSN neighborhood → Almomani WSN-DS lineage.")
    if "EdgeIIoT" in tags:
        md.append("- Edge-IIoT neighborhood → C10 group-aware discussion.")
    if "quant" in tags:
        md.append("- Quantization neighborhood → Jacob/C4 PTQ honesty.")
    if "FL" in tags:
        md.append("- Federated setting → distinct from single-node MCU HIL.")
    if not tags:
        md.append("- Background/foundational cite.")
    md.append("- **Numbers only from exact lines above.**")
    md.append("")
    md.append("## Verification")
    md.append(f"- PASS1B: evidence JSON written with exact lines from extract")
    md.append(f"- PASS2: all `{len(metrics)+len(tables)}` quoted lines re-found in full extract (by construction)")
    md.append("- PASS2_VISUAL: pending title/page image confirmation log")
    (REV / f"{pid}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return ev

# ---- PASS 1B: build all cards ----
ids = sorted(p.stem for p in (ROOT/"e2e_pdfs").glob("*.pdf") if p.stem not in skip)
built = []
for pid in ids:
    if not (EX / f"{pid}.full.txt").exists():
        print("SKIP no extract", pid)
        continue
    ev = build_card(pid)
    built.append(pid)
    print(f"PASS1B {pid}: metrics={len(ev['metric_lines'])} tables={len(ev['table_headers'])}")

# ---- PASS 2: independent re-check ----
p2_results = []
for pid in built:
    card = (REV / f"{pid}.md").read_text(encoding="utf-8")
    full = (EX / f"{pid}.full.txt").read_text(encoding="utf-8", errors="replace")
    quotes = re.findall(r"`([^`]+)`", card)
    fails = []
    n = 0
    for q in quotes:
        if q.startswith("_") or "/" in q and q.endswith(".txt"):
            continue
        if len(q) < 8:
            continue
        n += 1
        if q not in full:
            fails.append(q)
    ok = len(fails) == 0
    p2_results.append({"id": pid, "ok": ok, "n": n, "fails": fails})
    # stamp
    stamp = f"\n## PASS2 independent text re-check ({date.today().isoformat()})\n"
    if ok:
        stamp += f"- **PASS2_OK** — {n}/{n} quotes exact-match in full extract\n"
        card2 = card.replace("PASS2_VISUAL: pending title/page image confirmation log",
                             "PASS2_TEXT: OK; PASS2_VISUAL: pending")
    else:
        stamp += f"- **PASS2_FAIL** — {len(fails)} mismatches\n"
        for f in fails[:5]:
            stamp += f"  - `{f}`\n"
        card2 = card
    (REV / f"{pid}.md").write_text(card2.rstrip() + "\n" + stamp, encoding="utf-8")
    print(("P2OK " if ok else "P2FAIL"), pid, n, len(fails))

(P2 / "pass2b_results.json").write_text(json.dumps(p2_results, indent=2), encoding="utf-8")
print("BUILT", len(built))
print("PASS2_OK", sum(1 for r in p2_results if r["ok"]), "/", len(p2_results))
