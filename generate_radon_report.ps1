# generate_radon_report.ps1
# Regenerates radon report with Cyclomatic Complexity, Raw Metrics, and Halstead Metrics.
# Usage: .\generate_radon_report.ps1 [-OutFile <filename>]

param (
    [string]$OutFile = "radon_report.txt"
)

$target  = "memorymesh"

Write-Host "Generating radon report for '$target'..." -ForegroundColor Cyan

$cc  = python -m radon cc  $target -s -a 2>&1
$raw = python -m radon raw $target -s   2>&1
$hal = python -m radon hal $target      2>&1

@"
===== Cyclomatic Complexity =====
$cc

===== Raw Metrics =====
$raw

===== Halstead Metrics =====
$hal
"@ | Set-Content -Path $outFile -Encoding UTF8

Write-Host "Report written to $outFile" -ForegroundColor Green
