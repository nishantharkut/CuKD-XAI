import json, re
from pathlib import Path
from collections import defaultdict

# Build a dense chronological narrative source: first 120 chars of EVERY user msg + first agent reply after each "topic shift"
p = Path(r"C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl")

def extract_text(content):
    if content is None: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts=[]
        for c in content:
            if isinstance(c,str): parts.append(c)
            elif isinstance(c,dict):
                t=c.get("text") or c.get("content") or ""
                if isinstance(t,str): parts.append(t)
        return "\n".join(parts)
    return str(content)

dialogue=[]
for i,line in enumerate(p.open("r",encoding="utf-8",errors="replace"),1):
    line=line.strip()
    if not line: continue
    try: obj=json.loads(line)
    except: continue
    if obj.get("type")!="event_msg": continue
    payload=obj.get("payload") or {}
    et=payload.get("type")
    if et not in ("user_message","agent_message"): continue
    text=payload.get("message") or payload.get("text") or extract_text(payload.get("content"))
    if not isinstance(text,str): text=str(text)
    dialogue.append({"line":i,"ts":obj.get("timestamp",""),"role":"user" if et=="user_message" else "assistant","text":text,"n":len(text)})

# Create compact narrative: for each user message, attach next assistant message preview
pairs=[]
i=0
while i < len(dialogue):
    d=dialogue[i]
    if d["role"]=="user":
        # find next assistant
        j=i+1
        asst=None
        while j < len(dialogue) and dialogue[j]["role"]!="user":
            if dialogue[j]["role"]=="assistant" and asst is None:
                asst=dialogue[j]
            j+=1
        pairs.append({
            "user_ts": d["ts"],
            "user_chars": d["n"],
            "user": re.sub(r"\s+"," ", d["text"])[:300],
            "asst_ts": asst["ts"] if asst else None,
            "asst": re.sub(r"\s+"," ", asst["text"])[:220] if asst else None,
        })
    i+=1

# Write compact pair index
out=Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_codex_session_pairs.md")
with out.open("w",encoding="utf-8") as f:
    f.write(f"# User→Assistant pair index ({len(pairs)} pairs)\n\n")
    for n,pr in enumerate(pairs,1):
        f.write(f"### Pair {n} | {pr['user_ts']} | user {pr['user_chars']}c\n")
        f.write(f"**User:** {pr['user']}\n\n")
        f.write(f"**Assistant:** {pr['asst']}\n\n")

# Also extract key long user messages (chars>500) as they often set direction
longs=[(i+1,pr) for i,pr in enumerate(pairs) if pr["user_chars"]>500]
print("pairs", len(pairs), "long_user", len(longs))
print("wrote", out, out.stat().st_size)
# print long user first 40 previews of first 100 chars of long msgs for narrative
for n,pr in longs[:25]:
    print(f"LONG#{n} {pr['user_ts'][:16]} {pr['user_chars']}c | {pr['user'][:140]}")
print("--- long mid ---")
mid=len(longs)//2
for n,pr in longs[mid:mid+15]:
    print(f"LONG#{n} {pr['user_ts'][:16]} {pr['user_chars']}c | {pr['user'][:140]}")
print("--- long end ---")
for n,pr in longs[-20:]:
    print(f"LONG#{n} {pr['user_ts'][:16]} {pr['user_chars']}c | {pr['user'][:140]}")
