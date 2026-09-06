@echo off
rem SPDX-License-Identifier: MPL-2.0
rem This Source Code Form is subject to the terms of the Mozilla Public
rem License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

setlocal
set "VSLANG=1033"
if not defined FCPE_BUILD_DIR set "FCPE_BUILD_DIR=build"
rem Invoke from the repository root. CMake 3.x can use VS 2026 through Ninja.
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "FCPE_VS=%%i"
if not defined FCPE_VS exit /b 1
call "%FCPE_VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cmake -S . -B "%FCPE_BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release %*
if errorlevel 1 exit /b 1
cmake --build "%FCPE_BUILD_DIR%" --parallel
exit /b %errorlevel%
