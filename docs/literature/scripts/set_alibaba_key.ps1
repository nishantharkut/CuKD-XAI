# Update Alibaba Coding Plan API key for Grok / Codex / Claude Code (User env).
# Usage:
#   .\docs\literature\scripts\set_alibaba_key.ps1 -ApiKey "sk-sp-..."
# Then open a NEW terminal before launching codex/claude.

param(
  [Parameter(Mandatory = $true)]
  [string]$ApiKey
)

[Environment]::SetEnvironmentVariable('ALIBABA_CODING_PLAN_API_KEY', $ApiKey, 'User')
[Environment]::SetEnvironmentVariable('DASHSCOPE_API_KEY', $ApiKey, 'User')
[Environment]::SetEnvironmentVariable('ANTHROPIC_AUTH_TOKEN', $ApiKey, 'User')
[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL', 'https://coding-intl.dashscope.aliyuncs.com/apps/anthropic', 'User')

Write-Host "Updated User env:"
Write-Host "  ALIBABA_CODING_PLAN_API_KEY (len=$($ApiKey.Length))"
Write-Host "  ANTHROPIC_AUTH_TOKEN (same)"
Write-Host "  ANTHROPIC_BASE_URL=https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"
Write-Host "Open a NEW PowerShell, then:"
Write-Host '  codex -C "C:\N Drive\Research\Cukd-XAI\CuKD-XAI" --profile alibaba-qwen'
Write-Host '  claude --model qwen3-coder-plus'
