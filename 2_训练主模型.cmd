@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 2 : train the main model  (mpchain)
echo ================================================
echo.
echo Config: N=40 d=128 dmp=128 T=6 batch=128 lr=1e-3  100k steps
echo One seed takes about 3h on an RTX PRO 6000, longer on a laptop GPU.
echo Press Ctrl+C to stop at any time; the last checkpoint stays on disk.
echo.
set SEED=%1
if "%SEED%"=="" set SEED=0
echo seed = %SEED%
echo.
python train_tr.py --steps 100000 --N 40 --d 128 --dmp 128 --T 6 ^
  --batch 128 --lr 1e-3 --eval 5000 --seed %SEED% --ckpt ck_seed%SEED%.pt
if errorlevel 1 goto fail
echo.
echo [OK] ck_seed%SEED%.pt written.
echo Run this file with 0 / 1 / 2 to get the three seeds used in the paper.
pause
exit /b 0

:fail
echo.
echo [ERROR] training failed, see the message above.
pause
exit /b 1
