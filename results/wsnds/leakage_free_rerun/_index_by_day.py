import json
import re
from pathlib import Path
from collections import defaultdict

idx = json.loads(Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_codex_session_index.json").read_text(encoding="utf-8"))
users = idx["user_message_previews"]

# group by date
by_day = defaultdict(list)
for u in users:
    day = (u["ts"] or "")[:10]
    by_day[day].append(u)

# write day-index with all previews
lines = ["# Codex session user-message index by day\n"]
lines.append(f"Total user messages: {len(users)}\n")
for day in sorted(by_day):
    msgs = by_day[day]
    lines.append(f"\n## {day} ({len(msgs)} user messages)\n")
    for u in msgs:
        lines.append(f"- **#{u['i']}** `{u['ts']}` ({u['chars']}c): {u['preview']}")

out = Path(r"C:\N Drive\Research\Cukd-XAI\CuKD-XAI\results\wsnds\leakage_free_rerun\_codex_session_user_index_by_day.md")
out.write_text("\n".join(lines), encoding="utf-8")
print("days", len(by_day))
for day in sorted(by_day):
    print(day, len(by_day[day]))
print("wrote", out, "bytes", out.stat().st_size)
