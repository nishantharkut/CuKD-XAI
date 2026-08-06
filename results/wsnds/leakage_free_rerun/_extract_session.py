import json
import re
from pathlib import Path
from datetime import datetime

p = Path(r"C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl")
out_dir = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun")
out_md = out_dir / "_codex_session_full_transcript.md"
out_users = out_dir / "_codex_session_user_messages.md"
out_index = out_dir / "_codex_session_index.json"

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
                t = c.get("text") or c.get("content") or c.get("input_text") or ""
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(t, list):
                    parts.append(extract_text(t))
        return "\n".join(parts)
    if isinstance(content, dict):
        return extract_text(content.get("text") or content.get("content") or "")
    return str(content)

user_msgs = []
agent_msgs = []
all_dialogue = []  # chronological user+agent only
tool_names = {}
fc_count = 0
patch_count = 0
web_count = 0
compact_count = 0
abort_count = 0
meta = {}

# Also track response_item messages by role for completeness
ri_user = 0
ri_assistant = 0

for i, line in enumerate(p.open("r", encoding="utf-8", errors="replace"), 1):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except Exception:
        continue
    ts = obj.get("timestamp", "")
    t = obj.get("type")
    payload = obj.get("payload") or {}

    if t == "session_meta":
        meta = {k: payload.get(k) for k in list(payload.keys())[:30]}
        continue

    if t == "event_msg":
        et = payload.get("type")
        if et == "user_message":
            text = payload.get("message") or payload.get("text") or extract_text(payload.get("content"))
            # some shapes
            if not text:
                text = extract_text(payload)
            entry = {"line": i, "ts": ts, "role": "user", "text": text if isinstance(text, str) else str(text)}
            user_msgs.append(entry)
            all_dialogue.append(entry)
        elif et == "agent_message":
            text = payload.get("message") or payload.get("text") or extract_text(payload.get("content"))
            entry = {"line": i, "ts": ts, "role": "assistant", "text": text if isinstance(text, str) else str(text)}
            agent_msgs.append(entry)
            all_dialogue.append(entry)
        elif et == "context_compacted":
            compact_count += 1
        elif et == "turn_aborted":
            abort_count += 1
        elif et == "patch_apply_end":
            patch_count += 1
        elif et == "web_search_end":
            web_count += 1
        continue

    if t == "response_item":
        pt = payload.get("type")
        if pt == "message":
            role = payload.get("role") or "?"
            text = extract_text(payload.get("content"))
            if role == "user":
                ri_user += 1
            elif role == "assistant":
                ri_assistant += 1
            # Prefer event_msg for dialogue to avoid duplicates; skip here
        elif pt == "function_call":
            fc_count += 1
            name = payload.get("name") or payload.get("tool_name") or "unknown"
            tool_names[name] = tool_names.get(name, 0) + 1
        elif pt == "custom_tool_call":
            name = payload.get("name") or payload.get("tool_name") or "custom"
            tool_names[name] = tool_names.get(name, 0) + 1

# Write full transcript (may be large)
# Cap individual message display in full md at 50k chars with note
MAX_MSG = 50000

with out_md.open("w", encoding="utf-8") as f:
    f.write(f"# Codex session transcript\n\n")
    f.write(f"- session file: `{p}`\n")
    f.write(f"- user messages (event): {len(user_msgs)}\n")
    f.write(f"- agent messages (event): {len(agent_msgs)}\n")
    f.write(f"- dialogue turns combined: {len(all_dialogue)}\n")
    f.write(f"- function/tool calls: {fc_count}\n")
    f.write(f"- patches: {patch_count}, web searches: {web_count}, compactions: {compact_count}, aborts: {abort_count}\n")
    f.write(f"- tool name histogram: {json.dumps(tool_names, indent=2)}\n")
    f.write(f"- session_meta keys: {json.dumps(meta, default=str)[:2000]}\n\n")
    f.write("---\n\n")
    for idx, m in enumerate(all_dialogue, 1):
        role = m["role"].upper()
        text = m["text"] or ""
        truncated = False
        if len(text) > MAX_MSG:
            text = text[:MAX_MSG] + f"\n\n...[truncated {len(m['text'])-MAX_MSG} chars]..."
            truncated = True
        f.write(f"## [{idx}] {role} @ {m['ts']} (jsonl line {m['line']})\n\n")
        f.write(text)
        f.write("\n\n---\n\n")

with out_users.open("w", encoding="utf-8") as f:
    f.write(f"# User messages only ({len(user_msgs)})\n\n")
    for idx, m in enumerate(user_msgs, 1):
        text = m["text"] or ""
        if len(text) > MAX_MSG:
            text = text[:MAX_MSG] + f"\n\n...[truncated]..."
        f.write(f"## [{idx}] @ {m['ts']} (line {m['line']})\n\n{text}\n\n---\n\n")

# Build phase-oriented index: first 80 chars of each user msg
index = {
    "session_path": str(p),
    "n_user": len(user_msgs),
    "n_agent": len(agent_msgs),
    "n_dialogue": len(all_dialogue),
    "fc_count": fc_count,
    "tool_names": tool_names,
    "patch_count": patch_count,
    "web_count": web_count,
    "compact_count": compact_count,
    "abort_count": abort_count,
    "ri_user_messages": ri_user,
    "ri_assistant_messages": ri_assistant,
    "user_message_previews": [
        {"i": i+1, "ts": m["ts"], "line": m["line"], "chars": len(m["text"] or ""), "preview": re.sub(r"\s+", " ", (m["text"] or ""))[:200]}
        for i, m in enumerate(user_msgs)
    ],
    "agent_message_previews": [
        {"i": i+1, "ts": m["ts"], "line": m["line"], "chars": len(m["text"] or ""), "preview": re.sub(r"\s+", " ", (m["text"] or ""))[:160]}
        for i, m in enumerate(agent_msgs)
    ],
    "outputs": {
        "full_transcript": str(out_md),
        "user_only": str(out_users),
    }
}
out_index.write_text(json.dumps(index, indent=2), encoding="utf-8")
print("users", len(user_msgs), "agents", len(agent_msgs), "dialogue", len(all_dialogue))
print("full md bytes", out_md.stat().st_size)
print("user md bytes", out_users.stat().st_size)
print("tools", tool_names)
print("first_user_ts", user_msgs[0]["ts"] if user_msgs else None)
print("last_user_ts", user_msgs[-1]["ts"] if user_msgs else None)
print("last_agent_ts", agent_msgs[-1]["ts"] if agent_msgs else None)
print("last 15 user previews:")
for m in user_msgs[-15:]:
    prev = re.sub(r"\s+", " ", (m["text"] or ""))[:140]
    print(f"  {m['ts']} | {prev}")
