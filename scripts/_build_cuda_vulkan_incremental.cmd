@echo off
REM Incremental CUDA+Vulkan build (no clean) of llama-server+llama-lab in the existing build dir.
REM Usage: scripts\_build_cuda_vulkan_incremental.cmd
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
rem --- expose newest Vulkan SDK ---
set "VULKAN_SDK="
for /f "delims=" %%V in ('dir /b /ad "C:\VulkanSDK" 2^>nul') do set "VULKAN_SDK=C:\VulkanSDK\%%V"
cd /d D:\_SATIN_AI_2\beellama
cmake --build build-x64-windows-cuda-vulkan-release --target llama-server llama-lab --parallel 24
exit /b %errorlevel%