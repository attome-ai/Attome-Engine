@echo off
setlocal enabledelayedexpansion

echo Starting GLSL shader compilation...

REM Compile all vertex shaders
for %%f in (*.vert) do (
    set "filename=%%~nf"
    set "output=!filename!_vert.spv"
    echo Compiling vertex shader: %%f -^> !output!
    glslc "%%f" -o "!output!"
)

REM Compile all fragment shaders
for %%f in (*.frag) do (
    set "filename=%%~nf"
    set "output=!filename!_frag.spv"
    echo Compiling fragment shader: %%f -^> !output!
    glslc "%%f" -o "!output!"
)

echo Compilation complete!
pause