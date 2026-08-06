import json
from pathlib import Path
from collections import Counter

p = Path(r"C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl")
roles = Counter()
item_types = Counter()
event_types = Counter()
# sample a few response_item and event_msg
samples = {"response_item": [], "event_msg": [], "turn_context": [], "compacted": [], "world_state": [], "session_meta": []}

for i, line in enumerate(p.open("r", encoding="utf-8", errors="replace"), 1):
    line = line.strip()
    if not line:
        continue
    obj = json.loads(line)
    t = obj.get("type")
    if t == "response_item":
        payload = obj.get("payload") or obj.get("item") or {}
        it = payload.get("type") or payload.get("role") or "?"
        item_types[it] += 1
        if len(samples["response_item"]) < 8:
            samples["response_item"].append({"line": i, "payload_type": it, "keys": list(payload.keys())[:20], "role": payload.get("role"), "content_type": type(payload.get("content")).__name__})
            # peek content structure
            c = payload.get("content")
            if isinstance(c, list) and c:
                samples["response_item"][-1]["content0_keys"] = list(c[0].keys()) if isinstance(c[0], dict) else type(c[0]).__name__
                if isinstance(c[0], dict):
                    samples["response_item"][-1]["content0_type"] = c[0].get("type")
                    txt = c[0].get("text") or c[0].get("content") or ""
                    if isinstance(txt, str):
                        samples["response_item"][-1]["content0_text_preview"] = txt[:200]
            elif isinstance(c, str):
                samples["response_item"][-1]["content_preview"] = c[:200]
    elif t == "event_msg":
        payload = obj.get("payload") or {}
        et = payload.get("type") or payload.get("msg_type") or "?"
        event_types[et] += 1
        if len(samples["event_msg"]) < 6:
            samples["event_msg"].append({"line": i, "etype": et, "keys": list(payload.keys())[:15]})
    elif t in samples and len(samples[t]) < 3:
        samples[t].append({"line": i, "keys": list(obj.keys()), "preview": {k: (str(v)[:150] if not isinstance(v, (dict, list)) else type(v).__name__) for k,v in list(obj.items())[:8]}})

out = {
    "item_types": item_types.most_common(),
    "event_types": event_types.most_common(40),
    "samples": samples,
}
Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_session_structure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps({"item_types": item_types.most_common(), "event_types": event_types.most_common(30)}, indent=2))
