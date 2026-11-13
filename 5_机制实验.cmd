@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 5 : mechanism experiments (routing)
echo ================================================
echo.
echo  (a) evaluation-time ablation : early exit, fixed depth, slot masking
echo  (b) route sensitivity        : one-character edits, 5 levels x 3 seeds
echo  (c) route shuffling          : swap routing plans between sentences
echo  (d) T=3 vs T=6 paired test
echo.
if not exist ck_seed0.pt (
  echo [SKIP] ck_seed0.pt not found - run 2_train_the_main_model first.
  pause
  exit /b 0
)
echo --- (a) evaluation-time ablation ---
python ablate_forced.py
if errorlevel 1 goto fail
echo.
echo --- (b) route sensitivity (takes a few minutes) ---
python route_v2.py --ckpts ck_seed0.pt ck_seed1.pt ck_seed2.pt --n 200
if errorlevel 1 goto fail
echo.
echo --- (c) route shuffling ---
python route_shuffle.py --ckpt ck_seed0.pt --n 500
if errorlevel 1 goto softfail
echo.
echo --- (d) T=3 vs T=6 ---
if exist ck_t3.pt python cmp_t3_t6.py
echo.
echo Done.
pause
exit /b 0

:softfail
echo.
echo [WARN] route_shuffle failed. It needs eval_bleu_any.py's data path
echo        and ck_seed*.pt; check the traceback above.
pause
exit /b 0

:fail
echo.
echo [ERROR] a mechanism experiment failed, see the message above.
pause
exit /b 1
