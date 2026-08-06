"""PASS2: every quoted metric/table snippet in review cards must appear in full extract."""
import re, json
from pathlib import Path
from datetime import date

ROOT = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\papers")
REV = ROOT / "reviews"
EX = ROOT / "_extract"
OUT = ROOT / "_pass2_verify"
OUT.mkdir(exist_ok=True)

results = []
for card in sorted(REV.glob("*.md")):
    pid = card.stem
    full_p = EX / f"{pid}.full.txt"
    if not full_p.exists():
        results.append({"id": pid, "ok": False, "error": "no full extract"})
        continue
    full = full_p.read_text(encoding="utf-8", errors="replace")
    # normalize whitespace for matching
    full_norm = re.sub(r"\s+", " ", full)
    text = card.read_text(encoding="utf-8")
    # quoted snippets in backticks on metric/table lines
    quotes = re.findall(r"`([^`]+)`", text)
    # only check longer snippets / metric-like
    fails = []
    checked = 0
    for q in quotes:
        q = q.strip()
        if len(q) < 12:
            continue
        if q.startswith("docs/") or q.startswith("_"):
            continue
        checked += 1
        qn = re.sub(r"\s+", " ", q)
        # allow partial: first 80 chars
        key = qn[:80]
        if key not in full_norm and qn not in full_norm:
            # try without punctuation variance
            key2 = re.sub(r"[^\w.%+\- ]", "", key)
            full2 = re.sub(r"[^\w.%+\- ]", " ", full_norm)
            full2 = re.sub(r"\s+", " ", full2)
            if key2[:60] not in full2:
                fails.append(q[:120])
    ok = len(fails) == 0
    rec = {"id": pid, "ok": ok, "n_checked": checked, "n_fail": len(fails), "fails": fails[:10]}
    results.append(rec)
    # append pass2 status to card
    stamp = f"\n## PASS2 text-verify ({date.today().isoformat()})\n"
    if ok:
        stamp += f"- **PASS2_TEXT_OK**: {checked} quoted snippets re-found in `_extract/{pid}.full.txt`\n"
    else:
        stamp += f"- **PASS2_TEXT_FAIL**: {len(fails)}/{checked} snippets not re-found\n"
        for f in fails[:5]:
            stamp += f"  - missing: `{f[:100]}`\n"
    # remove old pass2 section if present
    text2 = re.sub(r"\n## PASS2 text-verify.*", "", text, flags=re.S)
    # update status line
    if ok:
        text2 = text2.replace("PASS1_DONE (text extract + structured evidence; PASS2 pending)",
                              "PASS1_DONE + PASS2_TEXT_OK (image title check pending)")
    card.write_text(text2.rstrip() + "\n" + stamp, encoding="utf-8")
    print(("OK " if ok else "FAIL"), pid, f"checked={checked} fail={len(fails)}")

(OUT / "pass2_text_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
n_ok = sum(1 for r in results if r.get("ok"))
print(f"PASS2_TEXT: {n_ok}/{len(results)} OK")
