# Handoff for Claude Code — CuKD-XAI literature e2e

## Project root
```
C:\N Drive\Research\Cukd-XAI\CuKD-XAI
```

## Task
E2E literature review with **visual page review** (open PNGs, not text-only).
Target 30–40 papers. Write/update `docs/literature/E2E_LITERATURE_REVIEW.md`.

## Literature corpus (already on disk)
| Path | What |
|---|---|
| `docs/literature/papers/e2e_pdfs/` | 29 PDFs |
| `docs/literature/papers/e2e_pages/<id>/page_XXX.png` | ~504 page images |
| `docs/literature/papers/e2e_download_status.json` | download/raster status + failures |
| `docs/literature/papers/e2e_manifest.json` | target list |
| `docs/literature/ALIBABA_CODEX_CLAUDE_SETUP.md` | Alibaba setup notes |

## Research results (for positioning claims)
| Path | What |
|---|---|
| `results/leftover_e2e_closure/` | J, 5678, reseed, edge GA, deployment-style 10-seed |
| `results/paper_strength_e2e/06_claim_freeze.json` | C1–C10 / X1–X5 |
| `results/leftover_e2e_closure/FG_MEAN_RECONCILIATION.md` | FG mean 0.9141 |
| `results/leftover_e2e_closure/06_deployment_style_cached_soft_replication/` | 0.9485 reproduced; mean ~0.921 |

---

## Grok session files (this conversation)

**Session dir (primary):**
```
C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\
```

| File | Size (approx) | Use |
|---|---:|---|
| `compaction\segment_000.md` … `segment_002.md` | ~0.5 MB each | **Best human-readable prior context** (post-compaction) |
| `compaction\INDEX.md` | small | TOC for segments |
| `chat_history.jsonl` | ~0.06 MB | current live turns only (history was compacted) |
| `summary.json` | small | session metadata |
| `events.jsonl` | ~6.7 MB | event log — read on disk if needed |
| `updates.jsonl` | ~35 MB | full tool stream — **too large to paste** into chat |
| `rewind_points.jsonl` | ~1 MB | file snapshots |

Also related agent sessions under same parent folder (deep-research subagents):
```
C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\
  019fd2fb-2217-7d02-b59a-8395abf621fd\   ← this main Grok chat
  019fd6e4-f34f-7a62-b3e3-ec2acab00862\   ← deep-research related
  019fd6e2-e747-7102-a5cd-4792668607b9\
  019fd6e2-e746-7a83-baef-f10026598866\
  019fd6e1-25a7-7a22-a293-e8d400b4c68b\
```

Deep-research report:
```
C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\workflows\wf_019fd6e0a0ce70c1a6f2c4e56bb81c4f\scratch\report.md
```
(Note: that report was **text-only partial**, not full image e2e.)

**Practical tip:** Prefer this handoff MD + corpus paths + claim-freeze results over dumping multi-hundred-MB jsonl into Claude context.

---

## Codex session files

Codex stores rollouts as:
```
C:\Users\nhnis\.codex\sessions\YYYY\MM\DD\rollout-*.jsonl
```

**Primary CuKD-related Codex rollout (large ~160 MB):**
```
C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl
```
**Prefer the summary memo instead of the full jsonl:**
```
C:\Users\nhnis\.codex\memories\rollout_summaries\2026-05-27T05-29-25-bEkp-cukd_xai_complete_results_binder_v3.md
```

Other large historical rollouts (may not be lit-review-specific):
```
C:\Users\nhnis\.codex\sessions\2026\05\24\rollout-2026-05-24T12-24-50-019e59f1-f8fc-7f23-8dd3-e21d5751f7b1.jsonl   (~325 MB)
C:\Users\nhnis\.codex\sessions\2026\07\07\rollout-2026-07-07T17-50-38-019f3c85-ee74-7760-b9c4-754b038a0f0a.jsonl   (~354 MB)
```

Recent rollouts (Aug 2026; many ~30–115 MB under parallel workers):
```
C:\Users\nhnis\.codex\sessions\2026\08\04\
C:\Users\nhnis\.codex\sessions\2026\08\02\
```

Session index (paths + metadata only, small):
```
C:\Users\nhnis\.codex\session_index.jsonl
```

---

## What to tell Claude (paste)

```text
Continue CuKD-XAI literature review e2e.

1) Read project handoff (primary):
   C:\N Drive\Research\Cukd-XAI\CuKD-XAI\docs\literature\CLAUDE_HANDOFF_LIT_REVIEW.md

2) Grok prior context (readable MD segments, not huge jsonl):
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\compaction\INDEX.md
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\compaction\segment_000.md
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\compaction\segment_001.md
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\compaction\segment_002.md

3) Grok live session jsonl (optional; chat_history is tiny after compaction):
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\chat_history.jsonl
   C:\Users\nhnis\.grok\sessions\C%3A%5CN%20Drive%5CResearch%5CCukd-XAI%5CCuKD-XAI\019fd2fb-2217-7d02-b59a-8395abf621fd\updates.jsonl   (35MB — do not load whole file)

4) Codex CuKD binder (prefer memo over 160MB jsonl):
   C:\Users\nhnis\.codex\memories\rollout_summaries\2026-05-27T05-29-25-bEkp-cukd_xai_complete_results_binder_v3.md
   C:\Users\nhnis\.codex\sessions\2026\05\27\rollout-2026-05-27T05-29-25-019e67e8-b9ad-7d31-be0e-0f36f96a17e9.jsonl

Corpus:
- docs/literature/papers/e2e_pdfs/
- docs/literature/papers/e2e_pages/<paper>/page_*.png  ← OPEN IMAGES visually
- docs/literature/papers/e2e_download_status.json
- docs/literature/papers/e2e_manifest.json

Do NOT rely only on extracted txt. Review title page, method diagram, results tables for each paper.
Failed downloads → flag for manual PDF drop.
Write final review to docs/literature/E2E_LITERATURE_REVIEW.md
```

---

## Claude `/model` + Alibaba — why the full catalog is missing

**Expected behavior.** Claude Code’s `/model` picker is built around Anthropic-style slots (Opus / Sonnet / Haiku aliases), **not** a live DashScope model catalog. Routing to Alibaba does **not** auto-populate every Coding Plan model into that UI.

Your current `~/.claude/settings.json`:
- `model`: `qwen3-coder-plus`
- `ANTHROPIC_DEFAULT_SONNET_MODEL` → `qwen3.7-plus`
- `ANTHROPIC_DEFAULT_OPUS_MODEL` → `qwen3-coder-plus`
- `ANTHROPIC_DEFAULT_HAIKU_MODEL` → `qwen3-coder-next`
- base: `https://coding-intl.dashscope.aliyuncs.com/apps/anthropic`

So the picker may only show those **three alias slots** (or the single active model), not glm/kimi/minimax/etc.

### How to switch Alibaba models anyway

**Type the model id explicitly** (even if not listed):
```text
/model qwen3-coder-plus
/model qwen3.7-plus
/model qwen3.6-plus
/model qwen3.5-plus
/model qwen3-coder-next
/model glm-5
/model kimi-k2.5
/model MiniMax-M2.5
```

**Or restart Claude with the model flag:**
```bat
claude --bare --model qwen3.7-plus
```
or use:
```
C:\Users\nhnis\launch-claude-alibaba.cmd
```

**Or change defaults** in `%USERPROFILE%\.claude\settings.json` (`model` + the three `ANTHROPIC_DEFAULT_*_MODEL` env keys), then open a **new** Claude session.

Auth key (canonical, len 38 working key):
```
C:\Users\nhnis\.grok\.env.alibaba
```
