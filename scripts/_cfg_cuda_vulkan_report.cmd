@echo off
REM Configure-only for the CUDA+Vulkan preset; captures all warnings/diagnostics.
REM Usage: scripts\_cfg_cuda_vulkan_report.cmd
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
rem --- locate newest installed Vulkan SDK ---
set "VULKAN_SDK="
for /f "delims=" %%V in ('dir /b /ad "C:\VulkanSDK" 2^>nul') do set "VULKAN_SDK=C:\VulkanSDK\%%V"
if not defined VULKAN_SDK echo ERROR: Vulkan SDK not found under C:\VulkanSDK & exit /b 1
set "VULKAN_SDK=%VULKAN_SDK%"
cd /d D:\_SATIN_AI_2\beellama
if exist build-x64-windows-cuda-vulkan-release rmdir /s /q build-x64-windows-cuda-vulkan-release
cmake --preset x64-windows-cuda-vulkan-release
exit /b %errorlevel%