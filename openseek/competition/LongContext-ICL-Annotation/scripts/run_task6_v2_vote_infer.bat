@echo off
setlocal
REM Task6 投票推理：优先用 Python（与 .py 同逻辑）；可选 Git Bash 跑 .bash

cd /d "%~dp0\.."

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python "%~dp0run_task6_v2_vote_infer.py" %*
  exit /b %ERRORLEVEL%
)
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 "%~dp0run_task6_v2_vote_infer.py" %*
  exit /b %ERRORLEVEL%
)

set "GITBASH=%ProgramFiles%\Git\bin\bash.exe"
if exist "%GITBASH%" (
  "%GITBASH%" "%~dp0run_task6_v2_vote_infer.bash" %*
  exit /b %ERRORLEVEL%
)

set "GITBASH32=%ProgramFiles(x86)%\Git\bin\bash.exe"
if exist "%GITBASH32%" (
  "%GITBASH32%" "%~dp0run_task6_v2_vote_infer.bash" %*
  exit /b %ERRORLEVEL%
)

where bash >nul 2>&1
if %ERRORLEVEL%==0 (
  bash "%~dp0run_task6_v2_vote_infer.bash" %*
  exit /b %ERRORLEVEL%
)

echo [错误] 未找到 python 或 bash。请安装 Python 或 Git for Windows。
exit /b 1
