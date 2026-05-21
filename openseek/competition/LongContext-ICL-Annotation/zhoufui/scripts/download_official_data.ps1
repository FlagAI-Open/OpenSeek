$ErrorActionPreference = "Stop"

$outDir = Join-Path $PSScriptRoot "..\data\official"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$base = "https://raw.githubusercontent.com/FlagAI-Open/OpenSeek/main/openseek/competition/LongContext-ICL-Annotation/data"
$files = @(
  "openseek-1_closest_integers.json",
  "openseek-2_count_nouns_verbs.json",
  "openseek-3_collatz_conjecture.json",
  "openseek-4_conala_concat_strings.json",
  "openseek-5_semeval_2018_task1_tweet_sadness_detection.json",
  "openseek-6_mnli_same_genre_classification.json",
  "openseek-7_jeopardy_answer_generation_all.json",
  "openseek-8_kernel_generation.json"
)

foreach ($file in $files) {
  $url = "$base/$file"
  $target = Join-Path $outDir $file
  Write-Host "Downloading $file"
  Invoke-WebRequest -Uri $url -OutFile $target
}

Get-ChildItem $outDir | Select-Object Name,Length
