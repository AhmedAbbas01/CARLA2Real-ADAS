param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$weightsDir = Join-Path $repoRoot 'weights'

function Find-Weight([string]$pattern) {
    $matches = Get-ChildItem -Path $weightsDir -File | Where-Object {
        $_.Name -match $pattern
    }

    if ($matches) {
        return $matches[0].FullName
    }

    return $null
}

$yoloModel = $env:YOLO_MODEL
if ([string]::IsNullOrWhiteSpace($yoloModel)) {
    $yoloModel = Find-Weight 'yolo'
}

$rtdetrModel = $env:RTDETR_MODEL
if ([string]::IsNullOrWhiteSpace($rtdetrModel)) {
    $rtdetrModel = Find-Weight 'rtdetr|rt-detr'
}

$fasterModel = $env:FASTER_MODEL
if ([string]::IsNullOrWhiteSpace($fasterModel)) {
    $fasterModel = Find-Weight 'faster|rcnn'
}

$depthModel = $env:DEPTH_MODEL
if ([string]::IsNullOrWhiteSpace($depthModel)) {
    $checkpointDir = Join-Path $repoRoot 'DepthAnythingV2/checkpoints'
    if (Test-Path $checkpointDir) {
        $candidate = Get-ChildItem -Path $checkpointDir -File | Where-Object { $_.Name -like 'depth_anything_v2*.pth' -or $_.Name -like 'depth_anything_v2*.pt' } | Select-Object -First 1
        if ($candidate) {
            $depthModel = $candidate.FullName
        }
    }
    if ([string]::IsNullOrWhiteSpace($depthModel)) {
        $depthModel = Join-Path $repoRoot 'DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    }
}

if (-not $yoloModel -or -not $rtdetrModel -or -not $fasterModel) {
    Write-Error "Missing one or more weights files in $weightsDir. Expected YOLO .pt, RT-DETR .pt, Faster R-CNN .pth."
    exit 1
}

Set-Location $scriptDir

$pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = (Get-Command py -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    $pythonExe = 'python'
}

if ($Args -contains '--help' -or $Args -contains '-h') {
    & $pythonExe main.py --help
    exit 0
}

& $pythonExe main.py `
    --mode ensemble `
    --host localhost `
    --port 2000 `
    --log-level INFO `
    --yolo-model $yoloModel `
    --rtdetr-model $rtdetrModel `
    --faster-model $fasterModel `
    --depth-model $depthModel `
    --num-classes 9 `
    --conf-threshold 0.40 `
    --cruise-throttle 0.35 `
    --warning-distance 15.0 `
    --brake-distance 7.0 `
    --lane-width 3.5 `
    --max-speed 30.0 `
    @Args
