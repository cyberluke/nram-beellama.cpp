# Reusable MSVC + CUDA + Vulkan env raiser, mirroring _build_ext_full_windows.ps1
# exactly so incremental builds (sccache/ninja) work from a cold PowerShell.
# dot-source this into the session:  . .\scripts\_env_msvc_cuda_vulkan.ps1
# Then run ninja targets normally against a configured build dir.

$ProgressPreference = "SilentlyContinue"

# --- MSVC (Community on this host) ---
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item "env:$($matches[1])" -Value $matches[2] }
}
# Pin cl.exe (Hostx64\x64) on PATH so nvcc's host pass and ninja can find it.
$msvcTools = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if ($msvcTools) {
    $clBin = Join-Path $msvcTools.FullName "bin\Hostx64\x64"
    if (Test-Path (Join-Path $clBin "cl.exe")) { $env:PATH = "$clBin;$env:PATH" }
}

# --- CUDA 13.3 ---
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if (-not (Test-Path $cudaPath)) { throw "CUDA 13.3 not found at $cudaPath" }
$env:CUDA_PATH = $cudaPath
$env:PATH = "$cudaPath\bin;$env:PATH"

# --- Vulkan SDK ---
$vulkanVer = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $vulkanVer) { throw "No Vulkan SDK under C:\VulkanSDK" }
$vulkanRoot = Join-Path "C:\VulkanSDK" $vulkanVer.Name
$env:VULKAN_SDK = $vulkanRoot
$env:PATH = "$vulkanRoot\Bin;$env:PATH"

# --- OpenSSL ---
$opensslRoot = "C:\Program Files\OpenSSL-Win64"
if (Test-Path $opensslRoot) { $env:OPENSSL_ROOT_DIR = $opensslRoot }

# --- Ninja on PATH ---
$ninjaExe = (Get-Command ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $ninjaExe) { $ninjaExe = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }
if (-not (Test-Path $ninjaExe)) { throw "ninja.exe not found" }
$env:PATH = "$(Split-Path $ninjaExe);$env:PATH"

Write-Host "[env] MSVC + CUDA $($cudaPath) + Vulkan $vulkanRoot ready; cl pinned; ninja on PATH"