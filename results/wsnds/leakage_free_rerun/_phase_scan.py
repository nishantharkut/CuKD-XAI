import json
import re
from pathlib import Path

# Re-parse user messages fully for substantive unique requests (filter short/acks)
p = Path(r"C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl")

def extract_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                t = c.get("text") or c.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return str(content)

users = []
for i, line in enumerate(p.open("r", encoding="utf-8", errors="replace"), 1):
    line=line.strip()
    if not line: continue
    try:
        obj=json.loads(line)
    except: continue
    if obj.get("type")!="event_msg": continue
    payload=obj.get("payload") or {}
    if payload.get("type")!="user_message": continue
    text = payload.get("message") or payload.get("text") or extract_text(payload.get("content"))
    if not isinstance(text,str): text=str(text)
    users.append({"line":i,"ts":obj.get("timestamp",""),"text":text})

# Keyword phase tags
keywords = {
    "edge_iiot": re.compile(r"edge[- ]?iiot|EdgeIIoT|edgeiiot", re.I),
    "hardware_hil": re.compile(r"hil|esp32|arduino|raspberry|pi5|firmware|hardware", re.I),
    "wsnds": re.compile(r"wsnds|WSN-DS|wsn-ds|leakage", re.I),
    "publication": re.compile(r"paper|manuscript|publication|journal|reviewer|submission", re.I),
    "gemini_doc": re.compile(r"gemini|google\.com/document|docs\.google", re.I),
    "student_b": re.compile(r"student\s*b|8192|9999", re.I),
    "tier15": re.compile(r"tier\s*1\.?5|confirmation|duplicate|feature.?group", re.I),
    "winterfell": re.compile(r"winterfell|remote", re.I),
    "codistill": re.compile(r"codistill|co-distill", re.I),
    "shap": re.compile(r"\bshap\b|xai|explain", re.I),
}

counts = {k:0 for k in keywords}
tagged = {k:[] for k in keywords}
for u in users:
    for k,pat in keywords.items():
        if pat.search(u["text"]):
            counts[k]+=1
            if len(tagged[k]) < 8:
                tagged[k].append({"ts":u["ts"], "preview": re.sub(r"\s+"," ",u["text"])[:180]})

# Extract unique substantive user intents: messages with ? or imperative length>80
intents = []
for u in users:
    t = u["text"].strip()
    # skip pure paste dumps that are mostly command output (heuristic: starts with path prompts)
    if len(t) < 20:
        continue
    # keep messages that look like instructions
    if any(x in t.lower() for x in ["please", "what", "why", "how", "should", "we need", "run ", "start ", "fix ", "check ", "tell me", "think", "confused", "complete", "leftover", "student"]):
        intents.append({"ts":u["ts"], "chars":len(t), "preview": re.sub(r"\s+"," ", t)[:250]})

out = {
    "keyword_counts": counts,
    "keyword_samples": tagged,
    "n_users": len(users),
    "substantive_intent_count": len(intents),
    "first_30_intents": intents[:30],
    "last_40_intents": intents[-40:],
    "mid_sample": intents[len(intents)//2-10:len(intents)//2+10] if len(intents)>20 else intents,
}
Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_codex_session_phases.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(counts, indent=2))
print("intents", len(intents))
print("--- first 20 intents ---")
for x in intents[:20]:
    print(x["ts"][:16], x["preview"][:160])
print("--- mid 10 ---")
mid=intents[len(intents)//2-5:len(intents)//2+5]
for x in mid:
    print(x["ts"][:16], x["preview"][:160])
print("--- last 25 ---")
for x in intents[-25:]:
    print(x["ts"][:16], x["preview"][:160])
