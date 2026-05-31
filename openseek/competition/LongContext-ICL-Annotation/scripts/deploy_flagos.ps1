# Windows：启动 / 停止本地 FlagOS（FlagScale serve）
param(
    [ValidateSet("run", "stop")]
    [string]$Action = "run",
    [string]$FlagScaleDir = "",
    [string]$ConfigName = "llm_config_flagos"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $FlagScaleDir) {
    $FlagScaleDir = Join-Path $RepoRoot "FlagScale"
}

if (-not (Test-Path $FlagScaleDir)) {
    Write-Error "未找到 FlagScale: $FlagScaleDir`n请先: git clone https://github.com/FlagOpen/FlagScale.git `"$FlagScaleDir`""
}

$ConfigSrc = Join-Path $RepoRoot "configs\$ConfigName.yaml"
$ConfigLink = Join-Path (Split-Path $FlagScaleDir -Parent) "$ConfigName.yaml"
Copy-Item -Force $ConfigSrc $ConfigLink

Push-Location $FlagScaleDir
try {
    python run.py --config-path .. --config-name $ConfigName "action=$Action"
} finally {
    Pop-Location
}
