@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo  Step 4 : evaluate and compute the statistics
echo ================================================
echo.
echo Needs the checkpoints produced by step 2 / step 3 in this folder.
echo Results land in evaldump\*.jsonl  (one file per checkpoint) and on screen.
echo.
if not exist ck_seed0.pt (
  echo [SKIP] ck_seed0.pt not found - run 2_train_the_main_model first.
  pause
  exit /b 0
)
echo --- BLEU-2 / token accuracy (same 500 sentences for every model) ---
set LIST=
if exist ck_seed0.pt set LIST=%LIST% ck_seed0.pt
if exist ck_seed1.pt set LIST=%LIST% ck_seed1.pt
if exist ck_seed2.pt set LIST=%LIST% ck_seed2.pt
if exist ck_moe_s0.pt set LIST=%LIST% ck_moe_s0.pt
if exist ck_ab_nomp.pt set LIST=%LIST% ck_ab_nomp.pt
if exist ck_ab_lastonly.pt set LIST=%LIST% ck_ab_lastonly.pt
if exist ck_ab_randroute.pt set LIST=%LIST% ck_ab_randroute.pt
echo checkpoints:%LIST%
python eval_bleu_any.py%LIST%
if errorlevel 1 goto fail
echo.
echo --- dense baseline, identical protocol ---
if exist ck_tf_s0.pt python eval_tf_bleu.py ck_tf_s0.pt
echo.
echo --- both protocols with the same sentence-end cutoff ---
if exist data/opus_train.npz python eval_bleu_stop.py
echo.
echo --- cross-seed mean/std and paired deltas ---
python stats_seeds.py
echo.
echo --- ablation table with bootstrap CI ---
python stats_ablation.py
if errorlevel 1 goto softfail
echo.
echo Done. See evaldump\ for the per-sentence dumps.
pause
exit /b 0

:softfail
echo.
echo [WARN] stats_ablation needs the ablation arms (ck_ab_*.pt).
echo        Train them first, then run this file again.
pause
exit /b 0

:fail
echo.
echo [ERROR] evaluation failed, see the message above.
pause
exit /b 1
