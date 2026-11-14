@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
rem --- locate newest installed Vulkan SDK ---
set "VULKAN_SDK="
for /f "delims=" %%V in ('dir /b /ad "C:\VulkanSDK" 2^>nul') do set "VULKAN_SDK=C:\VulkanSDK\%%V"
if not defined VULKAN_SDK echo ERROR: Vulkan SDK not found under C:\VulkanSDK & exit /b 1
echo VULKAN_SDK=%VULKAN_SDK%
rem --- clean rebuild of the combined CUDA+Vulkan target ---
if exist build-x64-windows-cuda-vulkan-release rmdir /s /q build-x64-windows-cuda-vulkan-release
cmake --preset x64-windows-cuda-vulkan-release
if errorlevel 1 exit /b 1
cmake --build build-x64-windows-cuda-vulkan-release --target llama-server llama-lab -j 24
exit /b %errorlevel%