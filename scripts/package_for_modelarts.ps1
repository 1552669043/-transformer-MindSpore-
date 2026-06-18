param(
    [string]$Output = "semanticnew_pet_upload.zip"
)

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutPath = Join-Path $Root $Output
$Temp = Join-Path $Root "_modelarts_package"

if (Test-Path -LiteralPath $Temp) {
    Remove-Item -LiteralPath $Temp -Recurse -Force
}
New-Item -ItemType Directory -Path $Temp | Out-Null

$Include = @(
    "configs",
    "data\pet",
    "docs",
    "reports",
    "scripts",
    "src",
    "eval.py",
    "train.py",
    "README.md",
    "requirements-modelarts.txt"
)

foreach ($Item in $Include) {
    $Source = Join-Path $Root $Item
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $Temp -Recurse
    }
}

Get-ChildItem -Path $Temp -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $Temp -Recurse -File -Include "*.pyc" | Remove-Item -Force

if (Test-Path -LiteralPath $OutPath) {
    Remove-Item -LiteralPath $OutPath -Force
}

Compress-Archive -Path (Join-Path $Temp "*") -DestinationPath $OutPath -Force
Remove-Item -LiteralPath $Temp -Recurse -Force

Write-Output "Created: $OutPath"
