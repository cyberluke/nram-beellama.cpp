param(
    [string]$Target = "llama-server",
    [int]$Parallel = 12
)
# Reusable CUDA-only configure+build validation for this host (Community MSVC, CUDA 13.3,
# RTX 3090 / sm_86, default FA + KVarN pair matrices). Mirrors scripts/
# build-win-cuda-sm_86-default.ps1 intent but sources the Community vcvars present on this box
# (the packaged script hardcodes BuildTools which is not installed here). Invoke from repo root:
#   powershell -File scripts/_build_cuda_validate.ps1 [target]
$ErrorActionPreference = "Stop"
$ProgressPreference  = "SilentlyContinue"
$repoRoot = $PSScriptRoot | Split-Path -Parent

# --- Locate MSVC (Community on this host) ---
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
Write-Host "[env] sourcing $vcvars"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item "env:$($matches[1])" -Value $matches[2] }
}

# --- Locate CUDA 13.3 ---
$cudaPath = Join-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" "v13.3"
if (-not (Test-Path $cudaPath)) { throw "CUDA 13.3 not found at $cudaPath" }
$env:CUDA_PATH = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"
$nvcc = Join-Path $cudaPath "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "nvcc.exe not found at $nvcc" }

$ninjaExe = (Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $ninjaExe) { $ninjaExe = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }
if (-not (Test-Path $ninjaExe)) { throw "ninja.exe not found" }
$env:PATH = "$(Split-Path $ninjaExe);$env:PATH"

$buildDir = Join-Path $repoRoot "build-win-cuda-sm_86-default"
Write-Host "========================================"
Write-Host "CPU+validated CUDA build (sm_86, default KVarN pairs)"
Write-Host "CUDA: $($env:CUDA_PATH)  Arch: 86  Ninja: $ninjaExe"
Write-Host "Build: $buildDir"
Write-Host "Target: $Target  Jobs: $Parallel"
Write-Host "========================================"

if (-not (Test-Path (Join-Path $buildDir "build.ninja"))) {
    Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
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
        # Tests off: with GGML_BACKEND_DL=ON + shared libs, ggml-cpu is a MODULE and cannot
        # be linked into test targets (tests/CMakeLists.txt:211). We validate llama-server here.
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=86",
        "-DCUDAToolkit_ROOT=$cudaPath",
        "-DCMAKE_CUDA_COMPILER=$nvcc",
        "-DCMAKE_MAKE_PROGRAM=$ninjaExe"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] cmake configure failed ($LASTEXITCODE)"; exit 1 }
} else {
    Write-Host "`n[CONFIGURE] reusing existing $buildDir\build.ninja"
}

Write-Host "`n[BUILD] ninja $Target (-j $Parallel)"
& "$ninjaExe" -C $buildDir $Target -j $Parallel
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] build failed ($LASTEXITCODE)"; exit 1 }
Write-Host "[OK] $Target built: $buildDir\bin\$Target.exe"
exit 0