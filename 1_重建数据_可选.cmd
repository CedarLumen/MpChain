@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 1 : rebuild data/un_sub.npz  (optional)
echo ================================================
echo.
echo The repo already ships data/un_sub.npz, so you only need this
echo if you want to rebuild it from the raw UN corpus.
echo.
if not defined MPCHAIN_SRC (
  echo [SKIP] MPCHAIN_SRC is not set.
  echo        PowerShell :  $env:MPCHAIN_SRC="D:\corpus\un_pc_en-zh_0.parquet"
  echo        cmd        :  set MPCHAIN_SRC=D:\corpus\un_pc_en-zh_0.parquet
  echo        then run this file again.
  echo.
  pause
  exit /b 0
)
python data.py
if errorlevel 1 goto fail
echo.
echo [OK] data/un_sub.npz rebuilt.
echo.
echo For the second corpus (OPUS-100) set MPCHAIN_OPUS_SRC the same way
echo and run:  python data_opus.py
pause
exit /b 0

:fail
echo.
echo [ERROR] rebuild failed, see the message above.
pause
exit /b 1
