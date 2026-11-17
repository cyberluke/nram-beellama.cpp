param(
    [string]$Target = "llama-server",
    [int]$Parallel = 6
)
# Full-feature Windows build: CUDA 13.3 (RTX 4090 / sm_89), Vulkan, AVX2, OpenSSL,
# MTMD, unified server, KVarN CUDA kernels — a single llama-server binary with all
# relevant backends/features enabled. For a different arch change $CudaArch.
#   powershell -File scripts/_build_ext_full_windows.ps1 [target]
$ErrorActionPreference = "Stop"
$ProgressPreference  = "SilentlyContinue"
$repoRoot = $PSScriptRoot | Split-Path -Parent

# --- MSVC (Community on this host) ---
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
Write-Host "[env] sourcing $vcvars"
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item "env:$($matches[1])" -Value $matches[2] }
}
# nvcc's host pass launches cl.exe directly; vcvars setting via the captured "set" can drop it
# from child-process PATH under parallel ninja, so pin cl.exe (Hostx64\x64) on PATH explicitly.
$msvcTools = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($msvcTools) {
    $clBin = Join-Path $msvcTools.FullName "bin\Hostx64\x64"
    if (Test-Path (Join-Path $clBin "cl.exe")) {
        $env:PATH = "$clBin;$env:PATH"
        Write-Host "[env] cl.exe pinned: $clBin"
    }
}

# --- CUDA 13.3, sm_89 (RTX 4090) ---
$cudaPath = Join-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" "v13.3"
if (-not (Test-Path $cudaPath)) { throw "CUDA 13.3 not found at $cudaPath" }
$env:CUDA_PATH = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"
$CudaArch = "89"

# --- Vulkan SDK (custom location C:\VulkanSDK) ---
$vulkanVer = (Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1)
if (-not $vulkanVer) { throw "No Vulkan SDK under C:\VulkanSDK" }
$vulkanRoot = Join-Path "C:\VulkanSDK" $vulkanVer.Name
Write-Host "[env] Vulkan SDK: $vulkanRoot"
$env:VULKAN_SDK = $vulkanRoot
$env:PATH = "$vulkanRoot\Bin;$env:PATH"

# --- OpenSSL (choco location) ---
$opensslRoot = "C:\Program Files\OpenSSL-Win64"
if (Test-Path $opensslRoot) { $env:OPENSSL_ROOT_DIR = $opensslRoot; Write-Host "[env] OpenSSL: $opensslRoot" }

# --- Ninja (Community path first, else PATH) ---
$ninjaExe = (Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $ninjaExe) { $ninjaExe = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }
if (-not (Test-Path $ninjaExe)) { throw "ninja.exe not found" }
$env:PATH = "$(Split-Path $ninjaExe);$env:PATH"

# --- Rust/cargo for LLGuidance (structured output) ---
if (Test-Path "$env:USERPROFILE\.cargo\bin\cargo.exe") {
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    Write-Host "[env] cargo (LLGuidance): $env:USERPROFILE\.cargo\bin"
} else {
    Write-Host "[warn] cargo not found; LLGuidance disabled" -ForegroundColor Yellow
}

$buildDir = Join-Path $repoRoot "build-win-ext-full"
Write-Host "========================================"
Write-Host "BeeLlama.cpp Windows full-feature build"
Write-Host "CUDA: $cudaPath (arch sm_$CudaArch)  Vulkan: $vulkanRoot"
Write-Host "AVX2: on (default)  OpenSSL: generated HTTPS  MTMD: on  LLGuidance: on"
Write-Host "Build: $buildDir  Target: $Target  Jobs: $Parallel"
Write-Host "========================================"

if (-not (Test-Path (Join-Path $buildDir "build.ninja"))) {
    Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
    $cmakeArgs = @(
        "-S", $repoRoot, "-B", $buildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_KVARN=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
        "-DGGML_CUDA_CUB_3DOT2=ON",
        "-DGGML_VULKAN=ON",
        "-DGGML_NATIVE=OFF",
        "-DGGML_CCACHE=OFF",
        "-DGGML_BACKEND_DL=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_RPC=ON",
        "-DLLAMA_OPENSSL=ON",
        "-DLLAMA_BUILD_BORINGSSL=ON",
        "-DLLAMA_LLGUIDANCE=OFF",
        "-DLLAMA_BUILD_MTMD=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        # llama-server is generated under BUILD_EXAMPLES in this tree; examples must be ON
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
        "-DCUDAToolkit_ROOT=$cudaPath",
        "-DCMAKE_CUDA_COMPILER=$cudaPath\bin\nvcc.exe",
        "-DCMAKE_MAKE_PROGRAM=$ninjaExe"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] cmake configure failed ($LASTEXITCODE)"; exit 1 }
} else {
    Write-Host "`n[CONFIGURE] reusing existing $buildDir\build.ninja"
}

Write-Host "`n[BUILD] ninja $Target (-j $Parallel)"
# Route through the log-guard so .ninja_log/.ninja_deps are auto-backed up before every
# build (a truncated log otherwise forces a full cold rebuild; sccache is dead here).
$guard = Join-Path $repoRoot "scripts\ninja_log_guard.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File $guard -BuildDir $buildDir -Target $Target -Parallel $Parallel
if ($LASTEXITCODE -ne 0) { Write-Host "[FAIL] build failed ($LASTEXITCODE)"; exit 1 }
Write-Host "[OK] $Target built: $buildDir\bin\$Target.exe"
exit 0