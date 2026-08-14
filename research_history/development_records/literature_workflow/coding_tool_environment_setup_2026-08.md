# Alibaba Coding Plan on Codex + Claude Code

Same key Grok uses: User env `ALIBABA_CODING_PLAN_API_KEY`  
OpenAI-compat base: `https://coding-intl.dashscope.aliyuncs.com/v1`  
Anthropic-compat base (Claude Code): `https://coding-intl.dashscope.aliyuncs.com/apps/anthropic`

## User env vars (already set)

| Variable | Purpose |
|---|---|
| `ALIBABA_CODING_PLAN_API_KEY` | Grok + Codex |
| `ANTHROPIC_BASE_URL` | Claude Code → Alibaba Anthropic app endpoint |
| `ANTHROPIC_AUTH_TOKEN` | Same key as Alibaba (Claude Code auth) |
| `DASHSCOPE_API_KEY` | Alias for some tools |

**Important:** Open a **new** terminal after env changes so Codex/Claude see them.

## Codex

Config: `%USERPROFILE%\.codex\config.toml`

- Provider: `[model_providers.alibaba_coding]`
- Profiles: `alibaba-qwen`, `alibaba-qwen37`, `alibaba-glm5`

### Start on CuKD project

```powershell
cd "C:\N Drive\Research\Cukd-XAI\CuKD-XAI"
# New shell so User env is loaded
codex -C "C:\N Drive\Research\Cukd-XAI\CuKD-XAI" --profile alibaba-qwen
```

Or one-shot overrides:

```powershell
codex -C "C:\N Drive\Research\Cukd-XAI\CuKD-XAI" `
  -c model_provider="alibaba_coding" `
  -c model="qwen3-coder-plus"
```

### Codex limitation (important)

Your Codex is **0.145** and only accepts `wire_api = "responses"`.  
Alibaba **Coding Plan** exposes **Chat Completions** (`/v1/chat/completions`), not the OpenAI **Responses** API.

So:

| Tool | Alibaba path | Status |
|---|---|---|
| **Grok** | OpenAI chat compat `/v1` | Works (when key valid) |
| **Claude Code** | Anthropic-compat `/apps/anthropic` | Preferred Alibaba path |
| **Codex** | needs Responses API | **May fail** until key works *and* Alibaba/gateway supports Responses |

Practical: use **Claude Code + Grok** for Alibaba models; keep Codex on OpenAI login for GPT models, or put a Responses-compatible proxy in front of DashScope.

### API key smoke test (2026-08-06)

User env `ALIBABA_CODING_PLAN_API_KEY` is set (len 22, `sk-sp-…`) but DashScope returned **401 invalid/expired**.  
Refresh the key in Alibaba Coding Plan console, then:

```powershell
cd "C:\N Drive\Research\Cukd-XAI\CuKD-XAI"
.\docs\literature\scripts\set_alibaba_key.ps1 -ApiKey "sk-sp-YOUR_NEW_KEY"
# open NEW terminal
```

## Claude Code

Config: `%USERPROFILE%\.claude\settings.json` (base URL only; token from User env)

```powershell
cd "C:\N Drive\Research\Cukd-XAI\CuKD-XAI"
# New shell
claude
# or
claude --model qwen3-coder-plus
```

If Claude still hits Anthropic OAuth, force bare API mode:

```powershell
$env:ANTHROPIC_BASE_URL="https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"
$env:ANTHROPIC_AUTH_TOKEN=$env:ALIBABA_CODING_PLAN_API_KEY
claude --bare --model qwen3-coder-plus
```

## Handoff prompt (lit review)

```text
Project: C:\N Drive\Research\Cukd-XAI\CuKD-XAI
PDFs: docs/literature/papers/e2e_pdfs/
Page PNGs: docs/literature/papers/e2e_pages/<id>/page_XXX.png
Status: docs/literature/papers/e2e_download_status.json
Visually open page images (not text-only). Grow to 35-40 papers. Write docs/literature/E2E_LITERATURE_REVIEW.md
```

## Backups

- `.codex\config.toml.bak-alibaba-*`
- `.claude\settings.json.bak-alibaba-*`
