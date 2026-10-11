param(
    [string]$RunDir = "logs\rsl_rl\ur5e_binpick_teacher\2026-08-07_01-26-09",
    [int]$NumEnvs = 512,
    [int]$Episodes = 512,
    [int]$Seed = 42,
    [int]$TopK = 5,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$resolvedRun = (Resolve-Path -LiteralPath $RunDir).Path
$runName = Split-Path -Leaf $resolvedRun
$stateDir = Join-Path $root "Phase1\logs\eval_all_$runName"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$batchLog = Join-Path $stateDir "batch.log"
$ledger = Join-Path $stateDir "progress.csv"
if (-not (Test-Path -LiteralPath $ledger)) {
    '"checkpoint","status","exit_code","started_at","finished_at","eval_dir"' |
        Set-Content -Encoding UTF8 -LiteralPath $ledger
}

function Get-CompletedCheckpoints {
    $completed = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $evalRoot = Join-Path $root "Phase1\Result"
    if (-not (Test-Path -LiteralPath $evalRoot)) {
        return ,$completed
    }

    foreach ($dir in Get-ChildItem -LiteralPath $evalRoot -Directory -Filter "eval_*") {
        $summary = Join-Path $dir.FullName "summary.txt"
        if (-not (Test-Path -LiteralPath $summary)) {
            continue
        }
        $text = Get-Content -Encoding UTF8 -Raw -LiteralPath $summary
        if ($text -notlike "*$runName*") {
            continue
        }
        if ($text -match "checkpoint\s*:\s*(.+)") {
            [void]$completed.Add([IO.Path]::GetFileName($Matches[1].Trim()))
        }
    }
    return ,$completed
}

function Find-EvalDirForCheckpoint([string]$checkpointName) {
    $evalRoot = Join-Path $root "Phase1\Result"
    if (-not (Test-Path -LiteralPath $evalRoot)) {
        return ""
    }
    foreach ($dir in Get-ChildItem -LiteralPath $evalRoot -Directory -Filter "eval_*" |
            Sort-Object LastWriteTime -Descending) {
        $summary = Join-Path $dir.FullName "summary.txt"
        if (-not (Test-Path -LiteralPath $summary)) {
            continue
        }
        $text = Get-Content -Encoding UTF8 -Raw -LiteralPath $summary
        if ($text -like "*$runName*" -and $text -like "*$checkpointName*") {
            return $dir.FullName
        }
    }
    return ""
}

$checkpoints = @(
    Get-ChildItem -LiteralPath $resolvedRun -File -Filter "model_*.pt" |
        Sort-Object { [int]($_.BaseName -replace "model_", "") }
)
$allCheckpointCount = $checkpoints.Count
if ($TopK -gt 0) {
    $metricsPath = Join-Path $root "Phase1\Result\$runName\metrics.csv"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        throw "TopK 후보 선별에 필요한 metrics.csv가 없습니다: $metricsPath"
    }
    $scoreByIteration = @{}
    foreach ($row in Import-Csv -LiteralPath $metricsPath) {
        $scoreByIteration[[int]$row.iteration] = [double]$row.'Train/mean_reward'
    }
    $checkpoints = @(
        $checkpoints |
            Where-Object {
                $iteration = [int]($_.BaseName -replace "model_", "")
                $scoreByIteration.ContainsKey($iteration)
            } |
            Sort-Object {
                $iteration = [int]($_.BaseName -replace "model_", "")
                $scoreByIteration[$iteration]
            } -Descending |
            Select-Object -First $TopK
    )
    $candidateCsv = Join-Path $stateDir "candidates.csv"
    $checkpoints |
        ForEach-Object {
            $iteration = [int]($_.BaseName -replace "model_", "")
            [pscustomobject]@{
                checkpoint = $_.Name
                iteration = $iteration
                train_mean_reward = $scoreByIteration[$iteration]
            }
        } |
        Export-Csv -Encoding UTF8 -NoTypeInformation -LiteralPath $candidateCsv
}
$completed = Get-CompletedCheckpoints

$header = "[batch] run=$runName selected=$($checkpoints.Count)/$allCheckpointCount " +
    "completed_total=$($completed.Count) num_envs=$NumEnvs episodes=$Episodes seed=$Seed top_k=$TopK"
$header | Tee-Object -FilePath $batchLog -Append

foreach ($checkpoint in $checkpoints) {
    if ($completed.Contains($checkpoint.Name)) {
        "[batch] skip completed: $($checkpoint.Name)" |
            Tee-Object -FilePath $batchLog -Append
        continue
    }

    $started = Get-Date
    "[batch] START $($checkpoint.Name) @ $($started.ToString('s'))" |
        Tee-Object -FilePath $batchLog -Append

    & $Python "Phase1\BinPick_Eval.py" `
        "--checkpoint" $checkpoint.FullName `
        "--num_envs" $NumEnvs `
        "--n_episodes" $Episodes `
        "--seed" $Seed `
        "--headless" 2>&1 |
        Tee-Object -FilePath $batchLog -Append
    $exitCode = $LASTEXITCODE

    $finished = Get-Date
    $evalDir = Find-EvalDirForCheckpoint $checkpoint.Name
    $status = if ($exitCode -eq 0 -and $evalDir) { "completed" } else { "failed" }
    $row = [pscustomobject]@{
        checkpoint = $checkpoint.Name
        status = $status
        exit_code = $exitCode
        started_at = $started.ToString("s")
        finished_at = $finished.ToString("s")
        eval_dir = $evalDir
    }
    $row | Export-Csv -Encoding UTF8 -NoTypeInformation -Append -LiteralPath $ledger

    "[batch] $($status.ToUpper()) $($checkpoint.Name) exit=$exitCode eval=$evalDir" |
        Tee-Object -FilePath $batchLog -Append

    if ($status -eq "completed") {
        [void]$completed.Add($checkpoint.Name)
    }
}

"[batch] DONE selected=$($checkpoints.Count) completed_total=$($completed.Count) " +
    "@ $((Get-Date).ToString('s'))" |
    Tee-Object -FilePath $batchLog -Append
