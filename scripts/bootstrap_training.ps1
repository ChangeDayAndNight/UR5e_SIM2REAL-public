[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$Constraints = Join-Path $RepoRoot "requirements\constraints-isaac51.txt"
$Requirements = Join-Path $RepoRoot "requirements\training-isaac51.txt"
$ExpectedIsaacLabCommit = "37ddf626871758333d6ed89cf64ad702aef127d0"
$EnvName = "isaac51"
$IsaacLabDir = Join-Path $RepoRoot "external\IsaacLab"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "[bootstrap] > $Executable $($Arguments -join ' ')"
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "명령 실패(exit=$LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

function Resolve-CondaEnvPython {
    param(
        [Parameter(Mandatory = $true)][string]$CondaExecutable,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $raw = & $CondaExecutable env list --json
    if ($LASTEXITCODE -ne 0) {
        throw "conda env list 실패"
    }
    $data = ($raw | Out-String) | ConvertFrom-Json
    foreach ($envPath in $data.envs) {
        if ((Split-Path -Leaf $envPath) -eq $Name) {
            $candidate = Join-Path $envPath "python.exe"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }
    return $null
}

function Resolve-CondaExecutable {
    # Activated conda sets CONDA_EXE even when shell initialization did not add
    # conda to PATH. Prefer explicit/current-shell locations, then standard
    # per-user Windows installations.
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
        $candidates += $env:CONDA_EXE
    }
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
        $candidates += $command.Source
    }
    foreach ($root in @("miniconda3", "anaconda3", "miniforge3")) {
        $candidates += Join-Path $env:USERPROFILE "$root\Scripts\conda.exe"
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

if ($env:OS -ne "Windows_NT") {
    throw "이 bootstrap은 Windows 11 native 전용입니다."
}
$WindowsBuild = [System.Environment]::OSVersion.Version.Build
if ($WindowsBuild -lt 22000) {
    throw "Windows 11 build>=22000이 필요합니다. actual=$WindowsBuild"
}
if (-not (Test-Path -LiteralPath $Constraints -PathType Leaf) -or
    -not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    throw "requirements/ 학습 고정 파일이 없습니다."
}

$Conda = Resolve-CondaExecutable
if ($null -eq $Conda) {
    throw "conda를 찾을 수 없습니다. Miniconda/Anaconda 설치 후 새 PowerShell에서 다시 실행하세요."
}
$Python = Resolve-CondaEnvPython -CondaExecutable $Conda -Name $EnvName
if ($null -eq $Python) {
    Invoke-Native -Executable $Conda -Arguments @("create", "--yes", "--name", $EnvName, "python=3.11.15")
    $Python = Resolve-CondaEnvPython -CondaExecutable $Conda -Name $EnvName
}
if ($null -eq $Python) {
    throw "conda 환경 '$EnvName'의 python.exe를 찾지 못했습니다."
}
$PythonVersion = (& $Python -c "import platform; print(platform.python_version())" | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "conda 환경 '$EnvName'의 Python 버전을 확인하지 못했습니다."
}
if ($PythonVersion -ne "3.11.15") {
    throw "기존 conda 환경 '$EnvName'의 Python 버전이 다릅니다. expected=3.11.15 actual=$PythonVersion. 환경을 명시적으로 재생성한 뒤 다시 실행하세요."
}
Write-Host "[bootstrap] Python: $Python"

$IsaacLabDir = [IO.Path]::GetFullPath($IsaacLabDir)
if (Test-Path -LiteralPath $IsaacLabDir) {
    if (-not (Test-Path -LiteralPath (Join-Path $IsaacLabDir ".git"))) {
        throw "IsaacLabDir가 존재하지만 git checkout이 아닙니다: $IsaacLabDir"
    }
} else {
    $externalRoot = Split-Path -Parent $IsaacLabDir
    New-Item -ItemType Directory -Path $externalRoot -Force | Out-Null
    Invoke-Native -Executable "git" -Arguments @(
        "clone", "--branch", "v2.3.2", "--depth", "1",
        "https://github.com/isaac-sim/IsaacLab.git", $IsaacLabDir
    )
}

$actualCommit = (& git -C $IsaacLabDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedIsaacLabCommit) {
    throw "Isaac Lab checkout 불일치: expected=$ExpectedIsaacLabCommit actual=$actualCommit"
}
$actualTag = (& git -C $IsaacLabDir describe --tags --exact-match).Trim()
if ($LASTEXITCODE -ne 0 -or $actualTag -ne "v2.3.2") {
    throw "Isaac Lab tag 불일치: expected=v2.3.2 actual=$actualTag"
}
$dirty = (& git -C $IsaacLabDir status --porcelain | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not [string]::IsNullOrWhiteSpace($dirty)) {
    throw "Isaac Lab checkout이 clean하지 않습니다: $dirty"
}

$env:ISAACLAB_ENV_PY = $Python
$env:ISAACLAB_DIR = $IsaacLabDir
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    # GitHub Actions starts each step in a new process. Persist the exact
    # interpreter selected above; the next step still revalidates it through
    # scripts/training_env.py instead of reconstructing a conda path.
    "ISAACLAB_ENV_PY=$Python" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}

# flatdict 4.0.1은 setuptools>=82 isolated build에서 pkg_resources 부재로 깨진다.
# 설치 이후에도 재현 가능한 호환 build-tool 세트를 그대로 유지한다.
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "pip==25.3", "setuptools==81.0.0", "wheel==0.45.1"
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "flatdict==4.0.1"
)

Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0",
    "--index-url", "https://download.pytorch.org/whl/cu128"
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "--constraint", $Constraints,
    "isaacsim[all,extscache]==5.1.0",
    "--extra-index-url", "https://pypi.nvidia.com"
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "--constraint", $Constraints, "--requirement", $Requirements
)

$core = Join-Path $IsaacLabDir "source\isaaclab"
$assets = Join-Path $IsaacLabDir "source\isaaclab_assets"
$rl = (Join-Path $IsaacLabDir "source\isaaclab_rl") + "[rsl_rl]"
$tasks = Join-Path $IsaacLabDir "source\isaaclab_tasks"
$project = Join-Path $RepoRoot "source\ur5e_binpick"

# core만 --no-deps: v2.3.2의 starlette==0.49.1 메타데이터는 Isaac Sim 5.1/FastAPI가
# 요구하는 starlette 0.45.3과 양립 불가다. 필요한 core deps는 위 requirements에 고정했다.
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "--no-deps", "--editable", $core
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "--no-deps", "--editable", $assets
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "--constraint", $Constraints, "--editable", $rl
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "--constraint", $Constraints, "--editable", $tasks
)
Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation",
    "--no-deps", "--editable", $project
)

Invoke-Native -Executable $Python -Arguments @(
    "-m", "pip", "install", "--disable-pip-version-check",
    "setuptools==81.0.0", "wheel==0.45.1", "packaging==23.0"
)

Invoke-Native -Executable $Python -Arguments @(
    (Join-Path $RepoRoot "scripts\preflight_training.py")
)

Write-Host "[bootstrap] BOOTSTRAP_TRAINING_OK"
Write-Host "[bootstrap] smoke: powershell -ExecutionPolicy Bypass -File scripts/train.ps1 -Preset Teacher -Smoke"
