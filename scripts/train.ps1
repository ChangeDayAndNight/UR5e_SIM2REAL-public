[CmdletBinding()]
param(
    [ValidateSet("Teacher", "Vision", "Distill", "NuRec")]
    [string]$Preset = "Teacher",
    [switch]$Smoke,
    [Nullable[int]]$NumEnvs,
    [Nullable[int]]$MaxIterations,
    [Nullable[int]]$Seed,
    [string]$DistillFrom = "",
    [string]$NuRecFrom = "",
    [switch]$Headless,
    [switch]$SkipPostprocess,
    [switch]$RequireNuRec,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path

function Resolve-TrainingPython {
    $resolver = Join-Path $RepoRoot "scripts\training_env.py"
    $attempts = @()

    if (-not [string]::IsNullOrWhiteSpace($env:ISAACLAB_ENV_PY) -and
        (Test-Path -LiteralPath $env:ISAACLAB_ENV_PY -PathType Leaf)) {
        $attempts += ,@($env:ISAACLAB_ENV_PY)
    }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher -and -not [string]::IsNullOrWhiteSpace($launcher.Source)) {
        $attempts += ,@($launcher.Source, "-3")
    }
    $hostPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $hostPython -and -not [string]::IsNullOrWhiteSpace($hostPython.Source)) {
        $attempts += ,@($hostPython.Source)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE) -and
        (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
        $attempts += ,@($env:CONDA_EXE, "run", "--name", "isaac51", "python")
    }
    # bootstrap_training.ps1 may have run in a separate PowerShell process, so
    # its environment variables cannot be relied on here. Match its discovery
    # for standard conda installs that were not added to PATH.
    foreach ($rootName in @("miniconda3", "anaconda3", "miniforge3")) {
        $root = Join-Path $env:USERPROFILE $rootName
        $directPython = Join-Path $root "envs\isaac51\python.exe"
        if (Test-Path -LiteralPath $directPython -PathType Leaf) {
            $attempts += ,@($directPython)
        }
        $condaExe = Join-Path $root "Scripts\conda.exe"
        if (Test-Path -LiteralPath $condaExe -PathType Leaf) {
            $attempts += ,@($condaExe, "run", "--name", "isaac51", "python")
        }
    }

    foreach ($attempt in $attempts) {
        $executable = $attempt[0]
        $prefix = @($attempt | Select-Object -Skip 1)
        $output = @(& $executable @prefix $resolver --resolve-python 2>&1)
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        $lines = @($output | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
        if ($lines.Count -eq 1 -and (Test-Path -LiteralPath $lines[0] -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $lines[0]).Path
        }
    }
    throw "canonical isaac51 Python을 찾지 못했습니다. scripts/bootstrap_training.ps1을 먼저 실행하세요."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$RequiredSentinel = ""
    )
    Write-Host "[train] > $Python $($Arguments -join ' ')"
    if ([string]::IsNullOrWhiteSpace($RequiredSentinel)) {
        & $Python @Arguments
        $exitCode = $LASTEXITCODE
    } else {
        $captured = @(& $Python @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
        foreach ($line in $captured) {
            Write-Host $line
        }
    }
    if ($exitCode -ne 0) {
        throw "Python 명령 실패(exit=$exitCode): $($Arguments -join ' ')"
    }
    if (-not [string]::IsNullOrWhiteSpace($RequiredSentinel)) {
        $outputText = ($captured | Out-String)
        if (-not $outputText.Contains($RequiredSentinel)) {
            throw "Python command exited 0 without required sentinel: $RequiredSentinel"
        }
        Write-Host "[train] sentinel verified: $RequiredSentinel"
    }
}

if ($null -ne $ExtraArgs) {
    $allowedSwitchFlags = @("--video", "--export_io_descriptors", "--resume")
    $allowedValueFlags = @("--video_length", "--video_interval", "--run_name", "--load_run", "--checkpoint",
                           "--accept-teacher-commit")   # F95: Distill 전용, 채택 run의 정확한 40자 commit
    for ($i = 0; $i -lt $ExtraArgs.Count; $i++) {
        $extraArg = $ExtraArgs[$i]
        if ($extraArg -match '^-{1,2}skip[_-]?preflight(?:=|$)') {
            throw "preflight bypass is forbidden: $extraArg"
        }
        if ($extraArg -match '^-{1,2}(task|experiment[_-]?name|distill[_-]?from|nurec[_-]?finetune)(?:=|$)' -or
            $extraArg -match '^(task|agent\.experiment_name)=') {
            throw "preset-owned training identity cannot be overridden through ExtraArgs: $extraArg"
        }
        if ($Preset -ne "Teacher" -and
            $extraArg -match '^-{1,2}(resume|load[_-]?run|checkpoint)(?:=|$)') {
            throw "Vision/Distill warm-start flags are owned by the preset: $extraArg"
        }
        if ($allowedSwitchFlags -contains $extraArg) {
            continue
        }
        $valueFlag = $null
        foreach ($candidate in $allowedValueFlags) {
            if ($extraArg -eq $candidate -or $extraArg.StartsWith("$candidate=")) {
                $valueFlag = $candidate
                break
            }
        }
        if ($null -ne $valueFlag) {
            if ($extraArg -eq $valueFlag) {
                if ($i + 1 -ge $ExtraArgs.Count -or $ExtraArgs[$i + 1].StartsWith("--")) {
                    throw "ExtraArg requires a value: $valueFlag"
                }
                $i += 1
            } elseif ($extraArg.Length -eq $valueFlag.Length + 1) {
                throw "ExtraArg requires a non-empty value: $valueFlag"
            }
            continue
        }
        if ($extraArg -notmatch '^-') {
            throw "orphan value or Hydra override is forbidden: $extraArg"
        }
        if ($extraArg -match '^-') {
            throw "unsupported ExtraArg outside the sealed launcher allowlist: $extraArg"
        }
    }
}

if ($Smoke -and $Preset -ne "Teacher") {
    throw "현재 -Smoke preset은 fail-closed Phase0 + Teacher 16env/1iter 계약입니다."
}
if ($Smoke -and $null -ne $NumEnvs -and $NumEnvs -ne 16) {
    throw "-Smoke contract is exactly -NumEnvs 16. Use a non-Smoke launch for other sizes."
}
if ($Smoke -and $null -ne $MaxIterations -and $MaxIterations -ne 1) {
    throw "-Smoke contract is exactly -MaxIterations 1. Use a non-Smoke launch for longer runs."
}
if ($Smoke -and $null -ne $Seed -and $Seed -ne 42) {
    throw "-Smoke contract is exactly -Seed 42. Use a non-Smoke launch for another seed."
}
if ($Smoke -and $null -ne $ExtraArgs -and $ExtraArgs.Count -gt 0) {
    # A smoke run is an exact scratch Teacher launch. Forwarding even an
    # apparently harmless flag can change task/resume/checkpoint/run/video Hydra
    # semantics, so no free-form arguments are accepted.
    throw "-Smoke rejects all ExtraArgs; use a non-Smoke launch for custom arguments."
}

$Python = Resolve-TrainingPython
$env:ISAACLAB_ENV_PY = $Python
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

Push-Location $RepoRoot
try {
    $preflight = @("scripts\preflight_training.py")
    if ($RequireNuRec) {
        $preflight += "--require-nurec"
    }
    if ($Preset -eq "NuRec" -and -not $RequireNuRec) {
        $preflight += "--require-nurec"
    }
    # The public launcher never bypasses the full runtime/GPU/registry preflight.
    Invoke-Python -Arguments $preflight

    if ($Smoke) {
        $NumEnvs = 16
        $MaxIterations = 1
        $Seed = 42
        Invoke-Python -Arguments @("Phase0\smoke_test.py", "--headless", "--steps", "60") `
            -RequiredSentinel "UR5E_SMOKE_OK robot=ur5e_cfg joints=8"
    }

    # A 20k Distill run produces ~41 checkpoints at the sealed 500-iteration
    # cadence.  A one-iteration integration smoke may run on a smaller disk,
    # but every longer/implicit Distill launch must have 25 GiB free first.
    if ($Preset -eq "Distill" -and
        ($null -eq $MaxIterations -or $MaxIterations -gt 1)) {
        Invoke-Python -Arguments @(
            "scripts\check_training_disk.py",
            "--path", $RepoRoot,
            "--minimum-gib", "25"
        ) -RequiredSentinel "FREE_SPACE_OK"
    }

    $launcher = switch ($Preset) {
        "Teacher" { "Phase1\BinPick_Train.py" }
        "Vision" { "Phase2\VisionPick_Train.py" }
        "Distill" { "Phase2\VisionPick_Train.py" }
        "NuRec" { "Phase2\VisionPick_Train.py" }
    }
    $arguments = @($launcher)
    if ($Preset -eq "Distill") {
        if (-not [string]::IsNullOrWhiteSpace($DistillFrom)) {
            $arguments += @("--distill-from", $DistillFrom)
        } else {
            throw "-Preset Distill에는 -DistillFrom <Teacher model_*.pt>가 필요합니다."
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($DistillFrom)) {
        throw "-DistillFrom은 -Preset Distill에서만 사용할 수 있습니다."
    }
    if ($Preset -eq "NuRec") {
        if ([string]::IsNullOrWhiteSpace($NuRecFrom)) {
            throw "-Preset NuRec에는 -NuRecFrom <sealed student model_*.pt>가 필요합니다."
        }
        $arguments += @("--nurec-finetune", $NuRecFrom)
    } elseif (-not [string]::IsNullOrWhiteSpace($NuRecFrom)) {
        throw "-NuRecFrom은 -Preset NuRec에서만 사용할 수 있습니다."
    }
    if ($null -ne $NumEnvs) {
        $arguments += @("--num_envs", [string]$NumEnvs)
    }
    if ($null -ne $MaxIterations) {
        $arguments += @("--max_iterations", [string]$MaxIterations)
    }
    if ($null -ne $Seed) {
        $arguments += @("--seed", [string]$Seed)
    }
    # 프로젝트 규칙상 학습은 항상 headless. -Headless는 표준 CLI의 명시적 자기문서화 옵션이다.
    $arguments += "--headless"
    if ($Smoke -or $SkipPostprocess) {
        $arguments += "--skip-postprocess"
    }
    if ($null -ne $ExtraArgs) {
        $arguments += $ExtraArgs
    }
    Invoke-Python -Arguments $arguments
    Write-Host "[train] TRAIN_PRESET_OK preset=$Preset smoke=$([bool]$Smoke)"
} finally {
    Pop-Location
}
