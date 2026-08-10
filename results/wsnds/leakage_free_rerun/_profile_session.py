import json
from pathlib import Path
from collections import Counter

p = Path(r"C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl")
types = Counter()
nested = Counter()
sample_keys = Counter()
n = 0
first_by_type = {}
for line in p.open("r", encoding="utf-8", errors="replace"):
    n += 1
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        types["JSON_ERROR"] += 1
        continue
    t = obj.get("type") or obj.get("record_type") or obj.get("kind") or "NO_TYPE"
    types[t] += 1
    if t not in first_by_type:
        first_by_type[t] = {k: (type(v).__name__ if not isinstance(v, (str, int, float, bool, type(None))) else v if not isinstance(v, str) or len(v) < 120 else v[:120] + "...") for k, v in list(obj.items())[:12]}
    for k in obj.keys():
        sample_keys[k] += 1
    for key in ("payload", "item", "message", "event"):
        if key in obj and isinstance(obj[key], dict):
            nt = obj[key].get("type") or obj[key].get("role") or obj[key].get("kind")
            if nt:
                nested[f"{t}.{key}.{nt}"] += 1

out = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_session_profile.json")
out.write_text(json.dumps({
    "total_lines": n,
    "types": types.most_common(),
    "nested": nested.most_common(50),
    "keys": sample_keys.most_common(40),
    "first_by_type": first_by_type,
}, indent=2), encoding="utf-8")
print("wrote", out)
print("total", n)
for k,v in types.most_common(30):
    print(f"  {k}: {v}")
