@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 3 : train the two baselines
echo ================================================
echo.
echo  (a) token-level MoE   - same data / steps / batch / lr as the main model
echo  (b) dense Transformer - 12 layers, d=384, ~23M params
echo.
echo Each takes a few hours. You can close the window after the checkpoint appears.
echo.
set SEED=%1
if "%SEED%"=="" set SEED=0
echo seed = %SEED%
echo.
echo --- (a) MoE ---
python train_moe.py --steps 100000 --N 40 --d 128 --mult 16 ^
  --batch 128 --lr 1e-3 --eval 5000 --seed %SEED% --ckpt ck_moe_s%SEED%.pt
if errorlevel 1 goto fail
echo.
echo --- (b) dense Transformer ---
python train_tf.py --steps 100000 --batch 128 --lr 3e-4 --eval 5000 ^
  --seed %SEED% --ckpt ck_tf_s%SEED%.pt
if errorlevel 1 goto fail
echo.
echo [OK] ck_moe_s%SEED%.pt and ck_tf_s%SEED%.pt written.
pause
exit /b 0

:fail
echo.
echo [ERROR] a baseline failed, see the message above.
pause
exit /b 1
