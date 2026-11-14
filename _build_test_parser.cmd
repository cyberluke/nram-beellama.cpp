@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\_SATIN_AI_2\beellama
cmake --build build-x64-windows-cuda-vulkan-release --target test-chat-auto-parser -j 8
exit /b %errorlevel%