@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 6 : efficiency measurement
echo ================================================
echo.
echo  (a) FLOPs per token, both architectures, one accounting standard
echo  (b) wall-clock: eager vs static buffers vs compiled vs CUDA graph
echo.
python -c "import torch; assert torch.cuda.is_available(), 'need a CUDA GPU'"
if errorlevel 1 goto nogpu
echo.
echo --- (a) FLOPs ---
python flops_report.py
if errorlevel 1 goto fail
echo.
echo --- (b) wall-clock (needs ck_seed0.pt) ---
if not exist ck_seed0.pt (
  echo [SKIP] ck_seed0.pt not found - run 2_train_the_main_model first.
  pause
  exit /b 0
)
python bench_both.py --ckpt ck_seed0.pt --nt 32 --reps 10
if errorlevel 1 goto fail
echo.
echo Done. batch=1 latency is latency-bound: it will not improve on a faster GPU.
pause
exit /b 0

:nogpu
echo.
echo [ERROR] no CUDA GPU found. This step needs one.
pause
exit /b 1

:fail
echo.
echo [ERROR] benchmark failed, see the message above.
pause
exit /b 1
