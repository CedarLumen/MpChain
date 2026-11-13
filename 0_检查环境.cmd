@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 0 : environment check
echo ================================================
echo.
echo --- Python ---
python -c "import sys; print(sys.version)"
if errorlevel 1 goto nopy
echo.
echo --- PyTorch / CUDA ---
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (training will be slow)')"
echo.
echo --- Data files ---
python -c "import os; [print(('OK   ' if os.path.exists(p) else 'MISS '), p) for p in ['data/un_sub.npz','data/opus_train.npz']]"
echo.
echo --- Module import check ---
python -c "import mpchain_tr, train_tr, train_moe, train_tf, eval_bleu_any, route_v2, route_shuffle, bench_both; print('all modules import OK')"
echo.
echo Done.
pause
exit /b 0

:nopy
echo.
echo [ERROR] python not found in PATH. Install Python 3.10+ and tick "Add to PATH".
pause
exit /b 1
