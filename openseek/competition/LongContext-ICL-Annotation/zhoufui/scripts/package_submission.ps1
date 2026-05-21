param(
    [string]$OutputDir = "",
    [string]$SourceZipName = "flagos_source_package.zip"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $out = Join-Path $root "outputs"
} elseif ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $out = $OutputDir
} else {
    $out = Join-Path $root $OutputDir
}
New-Item -ItemType Directory -Force -Path $out | Out-Null

$sourceZip = Join-Path $out $SourceZipName
$staging = Join-Path $out "_source_package_staging"

if (Test-Path $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $staging | Out-Null

$include = @(
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "configs",
    "data/sample_eval.jsonl",
    "data/sample_train.jsonl",
    "data/sample_official",
    "docs/final_technical_report_zh.md",
    "docs/final_reproduction_guide_zh.md",
    "docs/flagscale_qwen3_setup.md",
    "docs/submission_checklist.md",
    "docs/wsl_python_inference.md",
    "docs/score_ledger.json",
    "scripts",
    "src",
    "tests"
)

foreach ($item in $include) {
    $src = Join-Path $root $item
    if (Test-Path $src) {
        $dst = Join-Path $staging $item
        $parent = Split-Path $dst -Parent
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    }
}

$reportDocs = Get-ChildItem -LiteralPath (Join-Path $root "docs") -File | Where-Object {
    $_.Name -like "*zhoufui.md" -or $_.Name -like "*zhoufui.pdf"
}
foreach ($srcItem in $reportDocs) {
    $dst = Join-Path (Join-Path $staging "docs") $srcItem.Name
    New-Item -ItemType Directory -Force -Path (Split-Path $dst -Parent) | Out-Null
    Copy-Item -LiteralPath $srcItem.FullName -Destination $dst -Force
}

Get-ChildItem -LiteralPath $staging -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $staging -Recurse -File -Filter "*.pyc" | Remove-Item -Force

if (Test-Path $sourceZip) {
    Remove-Item -LiteralPath $sourceZip -Force
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $staging,
    $sourceZip,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)
Remove-Item -LiteralPath $staging -Recurse -Force

Write-Host "Wrote source package: $sourceZip"
