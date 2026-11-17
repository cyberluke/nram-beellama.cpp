# ninja_log_guard.ps1 — run ninja with automatic backup/restore of the ninja analytics
# files (.ninja_log / .ninja_deps), so a truncated/corrupt log never forces a full cold
# rebuild again (sccache is effectively dead on this machine).
#
# Usage:
#   powershell -File scripts\ninja_log_guard.ps1 -BuildDir <dir> [-Target <t>] [-Parallel N] [-RestoreIfCorrupt]
#
# Behavior:
#   - Always snapshots a rotated backup (.1, .2) of .ninja_log/.ninja_deps BEFORE ninja runs.
#   - On -RestoreIfCorrupt, restores the newest backup that is not older than build.ninja
#     so ninja sees the full history instead of "recovering" from a truncated log.
#   - After a failed build, prints where the backups are for a surgical recovery.

param(
    [Parameter(Mandatory = $true)][string]$BuildDir,
    [string]$Target = "llama-server",
    [int]$Parallel = 6,
    [switch]$RestoreIfCorrupt
)

$ErrorActionPreference = "Stop"
$ProgressPreference     = "SilentlyContinue"
$ninjaBin = "C:\Python314\Scripts\ninja.exe"

if (-not (Test-Path (Join-Path $BuildDir "build.ninja"))) {
    throw "build.ninja not found under $BuildDir - not a ninja build dir"
}

$Log  = Join-Path $BuildDir ".ninja_log"
$Deps = Join-Path $BuildDir ".ninja_deps"
$Log1 = Join-Path $BuildDir ".ninja_log.good.1"
$Deps1= Join-Path $BuildDir ".ninja_deps.good.1"
$Log2 = Join-Path $BuildDir ".ninja_log.good.2"
$Deps2= Join-Path $BuildDir ".ninja_deps.good.2"
$BNin = Join-Path $BuildDir "build.ninja"

# --- restore known-good snapshot (before build) when a corrupted state is detected ---
if ($RestoreIfCorrupt) {
    $bninMtime = if (Test-Path $BNin) { (Get-Item $BNin).LastWriteTime } else { Get-Date 0 }
    foreach ($cand in @($Log2, $Log1)) {
        if (-not (Test-Path $cand)) { continue }
        $depsCand = $cand -replace '\.log\.good\.', '.deps.good.'
        if ((Test-Path $depsCand) -and (Get-Item $cand).LastWriteTime -ge $bninMtime) {
            Write-Host "[log-guard] restoring known-good ninja analytics: $cand"
            Copy-Item $cand     $Log  -Force
            Copy-Item $depsCand $Deps -Force
            break
        }
    }
}

# --- rotate backups: .2 <- .1 <- current ---
if (Test-Path $Log1) {
    Copy-Item $Log1 $Log2 -Force
    if (Test-Path $Deps1) { Copy-Item $Deps1 $Deps2 -Force }
}
if (Test-Path $Log) {
    Copy-Item $Log  $Log1 -Force
    if (Test-Path $Deps) { Copy-Item $Deps $Deps1 -Force }
}

Write-Host "[log-guard] ninja -C $BuildDir $Target (jobs=$Parallel)"
& $ninjaBin -C $BuildDir -j $Parallel $Target
$code = $LASTEXITCODE
Write-Host "[log-guard] ninja exit=$code"

if ($code -ne 0) {
    Write-Host "[log-guard] WARNING: build failed (exit $code). Backups preserved at:" -ForegroundColor Yellow
    Write-Host "   $Log1" -ForegroundColor Yellow
    Write-Host "   $Log2" -ForegroundColor Yellow
    Write-Host "  Re-run with -RestoreIfCorrupt for a surgical recovery." -ForegroundColor Yellow
}

exit $code