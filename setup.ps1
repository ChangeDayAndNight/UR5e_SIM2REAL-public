<#
.SYNOPSIS
  배포 PC 1회 셋업 — conda 환경 생성 → 의존성 설치 → 점검 런북 실행.

.DESCRIPTION
  저장소를 내려받은 뒤 이 스크립트 하나만 실행하면 실행 준비가 끝납니다.
  로봇·카메라가 연결되어 있지 않아도 됩니다(그 단계는 건너뜁니다).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup.ps1
  powershell -ExecutionPolicy Bypass -File setup.ps1 -WithHardware   # RTDE/RealSense 연결까지 확인
#>
[CmdletBinding()]
param(
    [string]$EnvName = "env_ur_real",
    [string]$PolicyDir = "exported_experimental\teacher_177c7eefed0f",
    [switch]$WithHardware,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"    # 한국어 콘솔(cp949)에서 출력이 깨지지 않도록

function Write-Step([string]$text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

# ── 1. conda 확인 ─────────────────────────────────────────────────────────────
Write-Step "1/4 conda 확인"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $conda) {
    foreach ($candidate in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
    )) {
        if (Test-Path $candidate) { $conda = Get-Item $candidate; break }
    }
}
if ($null -eq $conda) {
    throw "conda를 찾지 못했습니다. Miniconda를 먼저 설치하십시오: https://docs.conda.io/en/latest/miniconda.html"
}
$condaExe = $conda.Source
if (-not $condaExe) { $condaExe = $conda.FullName }
Write-Host "    conda = $condaExe"

# ── 2. 환경 생성 ──────────────────────────────────────────────────────────────
Write-Step "2/4 conda 환경 '$EnvName' (python 3.11)"
$envList = & $condaExe env list
if ($envList -match "^\s*$([regex]::Escape($EnvName))\s") {
    Write-Host "    이미 존재합니다 — 건너뜁니다."
} else {
    & $condaExe create -n $EnvName python=3.11 -y
    if ($LASTEXITCODE -ne 0) { throw "conda 환경 생성 실패" }
}

$envRoot = (& $condaExe run -n $EnvName python -c "import sys; print(sys.prefix)").Trim()
$py = Join-Path $envRoot "python.exe"
if (-not (Test-Path $py)) { throw "환경의 python.exe를 찾지 못했습니다: $py" }
Write-Host "    python = $py"

# ── 3. 의존성 설치 ────────────────────────────────────────────────────────────
Write-Step "3/4 의존성 설치 (torch-cpu · ur-rtde · pyrealsense2 · opencv)"
if ($SkipInstall) {
    Write-Host "    -SkipInstall 지정 — 건너뜁니다."
} else {
    $lock = "deployment\requirements-win11.lock.txt"
    $wheelhouse = "deployment\wheelhouse"
    if (Test-Path $wheelhouse) {
        # 오프라인 배포 PC: 동봉된 wheelhouse만 사용한다.
        & $py -m pip install --no-index --find-links $wheelhouse --require-hashes -r $lock
    } else {
        & $py -m pip install --require-hashes -r $lock
    }
    if ($LASTEXITCODE -ne 0) {
        throw "의존성 설치 실패 — 인터넷 연결 또는 deployment\wheelhouse 를 확인하십시오."
    }
}

# ── 3b. 현장 설정 파일 ────────────────────────────────────────────────────────
#   configs\real.local.yaml 은 로봇 IP·카메라 시리얼 같은 현장 값이라 저장소에 없다(.gitignore).
#   없으면 템플릿에서 만들어 두고, 채울 값은 런북이 알려준다.
$local = "configs\real.local.yaml"
if (-not (Test-Path $local)) {
    Copy-Item "configs\real.example.yaml" $local
    Write-Host "    $local 생성(템플릿 복사) — robot.ip, camera.serial 을 채우십시오." -ForegroundColor Yellow
}

# ── 4. 점검 런북 ──────────────────────────────────────────────────────────────
Write-Step "4/4 배포 점검 런북"
$runbookArgs = @("deployment\DeployPC_Runbook.py", "--policy-dir", $PolicyDir)
if ($WithHardware) { $runbookArgs += "--with-hardware" }
& $py @runbookArgs
$runbookExit = $LASTEXITCODE

Write-Host ""
Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
if ($runbookExit -eq 0) {
    Write-Host "셋업 완료 — 모든 점검을 통과했습니다." -ForegroundColor Green
} else {
    Write-Host "셋업은 끝났고, 런북이 남은 현장 작업을 알려줍니다(위 '할 일')." -ForegroundColor Yellow
    Write-Host "보통 남는 것: configs\real.local.yaml 의 카메라 시리얼, 캘리브레이션, 모션 인가." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "다음 단계 (모션 없음, 추론만):" -ForegroundColor Cyan
Write-Host "  conda activate $EnvName"
Write-Host "  `$env:PYTHONIOENCODING='utf-8'"
Write-Host "  python Phase4\Run_Task_Teacher.py --policy-dir $PolicyDir --dry-run --verbose-perception"
Write-Host ""
Write-Host "자세한 절차는 DEPLOY_GUIDE.md 를 읽으십시오." -ForegroundColor Cyan
exit $runbookExit
