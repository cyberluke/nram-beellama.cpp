@echo off
setlocal
rem ===========================================================================
rem  The Lab - Community build script (INCREMENTAL, no rmdir wipe).
rem  Builds llama-server AND kelvin-tui, using all 24 threads, then assembles
rem  a self-contained dist\ folder with every binary so nothing is ever lost.
rem ===========================================================================
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0

set "BD=build-x64-windows-cuda-vulkan-release"
set "DIST=dist-kelvin-lab"

rem ---- 1) configure once (incremental) --------------------------------
cmake --preset x64-windows-cuda-vulkan-release
if errorlevel 1 exit /b 1

rem ---- 2) build both targets ------------------------------------------
cmake --build %BD% --target llama-server kelvin-tui -j 24
if errorlevel 1 exit /b 1

rem ---- 3) assemble dist (copy, never delete) --------------------------
set "DISTDIR=%~dp0dist-kelvin-lab"
if not exist "%DISTDIR%" mkdir "%DISTDIR%"
cd /d "%~dp0%BD%\bin"

copy /Y llama-server.exe            "%DISTDIR%\" >nul
copy /Y llama-server-impl.dll       "%DISTDIR%\" >nul
copy /Y llama-common.dll            "%DISTDIR%\" >nul
copy /Y llama.dll                   "%DISTDIR%\" >nul
copy /Y ggml.dll                    "%DISTDIR%\" >nul
copy /Y ggml-base.dll               "%DISTDIR%\" >nul

copy /Y ggml-cuda.dll               "%DISTDIR%\" >nul
copy /Y ggml-vulkan.dll             "%DISTDIR%\" >nul
copy /Y mtmd.dll                    "%DISTDIR%\" >nul

copy /Y ggml-cpu-x64.dll            "%DISTDIR%\" >nul
copy /Y ggml-cpu-alderlake.dll      "%DISTDIR%\" >nul
copy /Y ggml-cpu-cannonlake.dll     "%DISTDIR%\" >nul
copy /Y ggml-cpu-cascadelake.dll    "%DISTDIR%\" >nul
copy /Y ggml-cpu-haswell.dll        "%DISTDIR%\" >nul
copy /Y ggml-cpu-icelake.dll        "%DISTDIR%\" >nul
copy /Y ggml-cpu-sandybridge.dll    "%DISTDIR%\" >nul
copy /Y ggml-cpu-skylakex.dll       "%DISTDIR%\" >nul
copy /Y ggml-cpu-sse42.dll          "%DISTDIR%\" >nul

copy /Y kelvin-tui.exe              "%DISTDIR%\" >nul

rem FTXUI shared libs (kelvin-tui runtime dependency)
copy /Y ftxui-component.dll         "%DISTDIR%\" >nul
copy /Y ftxui-dom.dll               "%DISTDIR%\" >nul
copy /Y ftxui-screen.dll            "%DISTDIR%\" >nul

cd /d %~dp0
echo.
echo ===========================================================================
echo  Deploy complete: %DISTDIR%
dir "%DISTDIR%" /b
echo ===========================================================================
endlocal
exit /b 0