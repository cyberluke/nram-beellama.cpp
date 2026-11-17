# build-gate-aaf.ps1 — incremental MSVC+CUDA build of the AAF zero-copy producer change.
#
# Why this exists: scripts/build-win-cuda-sm_86-default.ps1 hardcodes the
# BuildTools vcvars path which does not exist on this machine (the usable
# installation is 2022 Community). This runner sources the real vcvarsall.bat,
# configures a DEDICATED working build dir (never the deployment dir
# build-win-ext-full, which is read-only), and builds only `llama-server`
# (the producer change is CPU-side; no new CUDA kernels/quant pairs).
#
# Usage:
#   powershell -File scripts\build-gate-aaf.ps1 [-BuildName build-win-zero-copy]
#              [-Target llama-server] [-Parallel 6] [-SkipConfigure]
#
# Reuses the ninja_log_guard snapshots so a truncated .ninja_log never forces
# a full cold rebuild. After a green build the binary lives at
# <BuildName>\bin\llama-server.exe for E2E verification against real models.

param(
    [string]$BuildName = "build-win-zero-copy",
    [string]$BuildDir = "",
    [string]$Target = "llama-server",
    [int]$Parallel = 6,
    [switch]$SkipConfigure
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$repoRoot = Split-Path -Parent $PSScriptRoot

# Absolute build dir override: lets us build OUT-OF-SOURCE on another drive
# (e.g. C:) when the repo's data drive is full, without touching anything there.
if (-not $BuildDir) { $buildDir = Join-Path $repoRoot $BuildName } else { $buildDir = $BuildDir }

# --- Real vcvars (2022 Community) ---
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) { throw "vcvarsall.bat not found at $vcvars" }

$cudaVer  = "13.3"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVer"
if (-not (Test-Path $cudaPath)) { throw "CUDA $cudaVer not found at $cudaPath" }

$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $ninjaExe) { $ninjaExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }
if (-not (Test-Path $ninjaExe)) { throw "ninja.exe not found" }

# --- Activate MSVC env from vcvars (x64) persistently into this process ---
Write-Host "[ENV] sourcing $vcvars x64"
cmd /c "`"$vcvars`" x64 > nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$env:CUDA_PATH = $cudaPath
$env:PATH = "$(Split-Path -Parent $ninjaExe);$cudaPath\bin;$env:PATH"
$env:MSBUILDDISABLENODEREUSE = "1"

# Sanity: cl + nvcc on PATH (memory: builds fail from a bare shell without INCLUDE)
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe not on PATH after vcvars; INCLUDE env missing?" }

if (-not (Test-Path (Join-Path $buildDir "build.ninja")) -and -not $SkipConfigure) {
    Write-Host "[CONFIGURE] cmake -S $repoRoot -B $buildDir"
    $cmakeArgs = @(
        "-S", $repoRoot, "-B", $buildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CCACHE=OFF",
        "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
        "-DGGML_CUDA_KVARN=ON",
        "-DGGML_CUDA_CUB_3DOT2=ON",
        "-DGGML_NATIVE=OFF",
        "-DGGML_BACKEND_DL=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_RPC=ON",
        "-DLLAMA_BUILD_BORINGSSL=ON",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=86",
        "-DCUDAToolkit_ROOT=$cudaPath",
        "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe",
        "-DCMAKE_MAKE_PROGRAM=$ninjaExe"
    )
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & cmake @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (exit $LASTEXITCODE)" }
} elseif ($SkipConfigure) {
    Write-Host "[CONFIGURE] skipped (SkipConfigure); reusing $buildDir\build.ninja"
}

# --- Build llama-server through the ninja_log_guard (snapshot / restore analytics) ---
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "ninja_log_guard.ps1") `
    -BuildDir $buildDir -Target $Target -Parallel $Parallel
if ($LASTEXITCODE -ne 0) { throw "ninja build of $Target failed (exit $LASTEXITCODE)" }

$exe = Join-Path $buildDir "bin\$Target.exe"
if (Test-Path $exe) {
    Write-Host "[OK] $Target built -> $exe"
} else {
    Write-Host "[WARN] $exe not found (target may not produce that exact name)"
}
exit 0