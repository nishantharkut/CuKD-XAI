# Open a draw.io file (default: system_hil.drawio)
param(
  [string]$File = ""
)
$draw = Join-Path $env:LOCALAPPDATA "Programs\draw.io\draw.io.exe"
$dir = Resolve-Path (Join-Path $PSScriptRoot "..\drawio")
if (-not $File) {
  $File = Join-Path $dir "system_hil.drawio"
} elseif (-not [System.IO.Path]::IsPathRooted($File)) {
  $File = Join-Path $dir $File
}
if (-not (Test-Path $draw)) { throw "draw.io not installed: $draw" }
if (-not (Test-Path $File)) { throw "Diagram not found: $File" }
Start-Process -FilePath $draw -ArgumentList "`"$File`""
Write-Host "Opened $File"
