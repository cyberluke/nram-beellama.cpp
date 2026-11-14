@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if exist build-x64-windows-msvc-release rmdir /s /q build-x64-windows-msvc-release
cmake --preset x64-windows-msvc-release
if errorlevel 1 exit /b 1
cmake --build build-x64-windows-msvc-release --target llama-server llama-lab -j 8
exit /b %errorlevel%