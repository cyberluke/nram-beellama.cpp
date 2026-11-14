@echo off
setlocal
REM Reusable CPU-only configure+build validation (no CUDA/Vulkan, no ccache).
REM Invoke from the repo root: scripts\_build_cpu_validate.cmd [target]
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\_SATIN_AI_2\beellama

set TARGET=%1
if "%TARGET%"=="" set TARGET=llama-server

if not exist build-x64-windows-msvc-release\build.ninja (
    echo [cfg] configuring CPU build ...
    cmake -S . -B build-x64-windows-msvc-release^
        -DCMAKE_BUILD_TYPE=RelWithDebInfo^
        -DGGML_CCACHE=OFF
    if errorlevel 1 exit /b 1
)

echo [build] building %TARGET% ...
cmake --build build-x64-windows-msvc-release --target %TARGET% --parallel 8
exit /b %errorlevel%