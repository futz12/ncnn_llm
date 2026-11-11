@echo off
REM Build script for llm_ncnn_run (adjust paths for your environment)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "PATH=%USERPROFILE%\.conda\envs\dl312\Scripts;%PATH%"
cd /d "%~dp0build"
ninja llm_ncnn_run
echo NINJA_EXIT=%ERRORLEVEL%
