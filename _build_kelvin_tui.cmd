@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\_SATIN_AI_2\beellama
cmake --preset x64-windows-cuda-vulkan-release >nul
if errorlevel 1 exit /b 1
cmake --build build-x64-windows-cuda-vulkan-release --target kelvin-tui -j 8
exit /b %errorlevel%