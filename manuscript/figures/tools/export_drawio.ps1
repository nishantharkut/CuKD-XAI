# Export all .drawio diagrams under manuscript/figures/drawio to PDF + PNG
$ErrorActionPreference = "Stop"
$draw = Join-Path $env:LOCALAPPDATA "Programs\draw.io\draw.io.exe"
if (-not (Test-Path $draw)) {
  Write-Error "draw.io not found at $draw. Install: winget install JGraph.Draw"
}
$root = Resolve-Path (Join-Path $PSScriptRoot "..\drawio")
$export = Join-Path $root "export"
New-Item -ItemType Directory -Force -Path $export | Out-Null

Get-ChildItem $root -Filter "*.drawio" | ForEach-Object {
  $stem = $_.BaseName
  $pdf = Join-Path $export "$stem.pdf"
  $png = Join-Path $export "$stem.png"
  $svg = Join-Path $export "$stem.svg"
  Write-Host "Exporting $($_.Name) ..."
  & $draw --export --format pdf --output $pdf $_.FullName
  if ($LASTEXITCODE -ne 0) { Write-Warning "PDF export exit $LASTEXITCODE for $($_.Name)" }
  & $draw --export --format png --output $png $_.FullName
  if ($LASTEXITCODE -ne 0) { Write-Warning "PNG export exit $LASTEXITCODE for $($_.Name)" }
  & $draw --export --format svg --output $svg $_.FullName
  Write-Host "  -> $pdf"
}

Write-Host "Done. Exports in $export"
Get-ChildItem $export | Select-Object Name, Length | Format-Table -AutoSize
