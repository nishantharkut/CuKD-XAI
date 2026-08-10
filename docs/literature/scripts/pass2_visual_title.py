import json, re
from pathlib import Path
from datetime import date

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
EX = ROOT / "_extract"
REV = ROOT / "reviews"
EV2 = ROOT / "_pass1b_evidence"
OUT = ROOT / "_pass2_verify"
pages = ROOT / "e2e_pages"

log = []
for jp in sorted(EV2.glob("*.json")):
    if jp.name.startswith("_"):
        continue
    ev = json.loads(jp.read_text(encoding="utf-8"))
    pid = ev["id"]
    full = (EX / f"{pid}.full.txt").read_text(encoding="utf-8", errors="replace")
    p1m = re.search(r"===== PAGE 1 =====\s*(.*?)(?:===== PAGE 2 =====|$)", full, re.S)
    p1 = p1m.group(1) if p1m else ""
    title = ev.get("title") or ""
    # title words >=4 chars present in page1?
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", title)]
    words = [w for w in words if w not in {"with","from","that","this","using","based","into","for","and","the"}]
    if words:
        hit = sum(1 for w in words if w in p1.lower())
        ratio = hit / max(len(words), 1)
    else:
        ratio = 0.0
    png = pages / pid / "page_001.png"
    rec = {
        "id": pid,
        "title": title,
        "title_word_hit_ratio": round(ratio, 3),
        "page1_png_exists": png.exists(),
        "n_metrics": len(ev.get("metric_lines") or []),
        "n_table_headers": len(ev.get("table_headers") or []),
        "abstract_len": len(ev.get("abstract") or ""),
        "visual_title_ok": ratio >= 0.5 and png.exists(),
    }
    # stamp card
    card_p = REV / f"{pid}.md"
    card = card_p.read_text(encoding="utf-8")
    stamp = f"\n## PASS2 visual title check ({date.today().isoformat()})\n"
    stamp += f"- page_001.png exists: {png.exists()}\n"
    stamp += f"- title word hit ratio vs page-1 text: {ratio:.3f}\n"
    if rec["visual_title_ok"]:
        stamp += "- **PASS2_VISUAL_TITLE_OK** (title words supported by page-1 text + PNG on disk)\n"
        card = card.replace("PASS2_TEXT: OK; PASS2_VISUAL: pending", "PASS2_TEXT: OK; PASS2_VISUAL_TITLE: OK")
        card = card.replace("**Status:** PASS1B_DONE + PASS2_TEXT_OK", "**Status:** PASS1B+PASS2_TEXT+PASS2_VISUAL_TITLE OK")
    else:
        stamp += "- **PASS2_VISUAL_TITLE_REVIEW** needed (low title hit or missing PNG)\n"
    if "PASS2 visual title check" not in card:
        card_p.write_text(card.rstrip() + "\n" + stamp, encoding="utf-8")
    log.append(rec)
    print(f"{'VOK' if rec['visual_title_ok'] else 'VREV'} {pid} ratio={ratio:.2f} metrics={rec['n_metrics']}")

(OUT / "pass2_visual_title.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print("VISUAL_TITLE_OK", sum(1 for r in log if r["visual_title_ok"]), "/", len(log))
# weak ones
weak = [r for r in log if not r["visual_title_ok"] or r["n_metrics"]==0]
print("NEED_ATTENTION", len(weak))
for r in weak:
    print(" ", r["id"], "ratio", r["title_word_hit_ratio"], "metrics", r["n_metrics"], "abs", r["abstract_len"])
