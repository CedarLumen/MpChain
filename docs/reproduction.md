# Reproducing the reported numbers

Section numbers refer to the manuscript (`paper/MpChain_Neurocomputing_v2.pdf`). Every command is
run from the repository root with `PYTHONPATH=src`, because the scripts import each other by module
name and read the corpus from `data/` relative to the working directory.

## 0. Environment

```
pip install -r requirements.txt
export MPCHAIN_SRC=/path/to/un_pc_en-zh_0.parquet
export MPCHAIN_OUT=$PWD/data/un_sub.npz
PYTHONPATH=src python src/data.py          # writes data/un_sub.npz
```

Optional second corpus (§5.7):

```
export MPCHAIN_OPUS_SRC=/path/to/train-00000-of-00001.parquet
export MPCHAIN_OPUS_OUT=$PWD/data/opus_train.npz
PYTHONPATH=src python src/data_opus.py
```

## 1. Training

| Model | Paper section | Command |
|---|---|---|
| The chain | §5.1 | `PYTHONPATH=src python src/train_tr.py --steps 100000 --N 40 --d 128 --dmp 128 --T 6 --batch 128 --lr 1e-3 --eval 5000 --ckpt ck_seed0.pt --seed 0` |
| Mixture of experts | §5.1 | `PYTHONPATH=src python src/train_moe.py --steps 100000 --N 40 --d 128 --mult 16 --batch 128 --lr 1e-3 --eval 5000 --ckpt ck_moe.pt` |
| Dense Transformer | §5.6 | `PYTHONPATH=src python src/train_tf.py --steps 100000 --batch 128 --d 384 --layers 12 --lr 3e-4 --eval 2500 --ckpt ck_tf100k.pt` |

Three seeds for the chain are produced by repeating the first command with `--seed 1 --ckpt
ck_seed1.pt` and `--seed 2 --ckpt ck_seed2.pt`; `scripts/02_train_ours.sh` does exactly that.

## 2. Evaluation and statistics

| Paper section | Reports | Command | Output |
|---|---|---|---|
| §5.1 | BLEU-2 and token accuracy, chain against mixture of experts | `PYTHONPATH=src python src/eval_bleu_any.py ck_seed0.pt ck_seed1.pt ck_seed2.pt ck_moe.pt` | table on stdout, per-sentence records in `evaldump/` |
| §5.6 | The same for the dense baseline | `PYTHONPATH=src python src/eval_tf_bleu.py ck_tf100k.pt` | stdout |
| §5.1, §5.6 | Paired differences, 95 % intervals, sign counts across seeds | `PYTHONPATH=src python src/stats_seeds.py` | reads `evaldump/`, writes statistics to stdout |
| §5.2 | Removing the shared memory, random routing, last call only | `PYTHONPATH=src python src/train_tr.py --no_mp 1 --ckpt ck_ab_nomp.pt`, then `--rand_route 1`, then `--last_only 1`; evaluate each as above | `results/EVAL_UN.txt` |
| §5.4 | Contribution of individual memory slots | `PYTHONPATH=src python src/slot_ablation.py` | stdout |
| §5.5 | Sensitivity of routing to single-character edits | `PYTHONPATH=src python src/route_sensitivity.py --ckpt ck_seed0.pt --n 60` | stdout |
| §5.5 | Permutation control (usage preserved, content broken) | `PYTHONPATH=src python src/route_shuffle.py --ckpt ck_seed0.pt --n 500` | stdout |
| §3.5 | Re-instantiating trained weights with a longer chain | `PYTHONPATH=src python src/eval_extend_t.py` | `results/EXTEND_T.txt` |
| §5.7 | Second corpus | repeat training and evaluation with `MPCHAIN_DATA=data/opus_train.npz` | `results/EVAL_OPUS.txt` |

## 3. Efficiency

| Paper section | Reports | Command | Output |
|---|---|---|---|
| §6.1 | FLOPs per token | `PYTHONPATH=src python src/flops_report.py` | `results/flops.txt` |
| §6.2 | Wall-clock along an optimisation ladder | `PYTHONPATH=src python src/bench_infer2.py --ckpt ck_seed0.pt --reps 20` | stdout |
| §6.2, §6.4 | Both architectures at the same optimisation level | `PYTHONPATH=src python src/bench_both.py --ckpt ck_seed0.pt --reps 15` | `results/BENCH_BOTH.txt` |
| §6.4 | Laptop GPU and CPU: latency, board power, energy | `PYTHONPATH=src python src/bench_edge.py --ckpt ck_seed0.pt --runs 3` | `results/EDGE_LOCAL.txt` |
| §6.4 | Raspberry Pi 5: latency, peak resident memory | `MPCHAIN_DATA=data/pi_sub.npz python src/pi_bench.py --threads 4` (and `--threads 1`) | `reports/PI_LOCAL.txt` next to the script |

`src/make_pi_data.py` produces the reduced archive the board needs: 600 test sentences and the
unchanged vocabularies, 0.04 MB instead of 20.5 MB, so the whole benchmark fits in 2 GB of RAM.

## 4. What the logs in results/ are

Every file under `results/` was written by one of the commands above on the machine named inside it;
none was edited by hand. `results/derived2.txt` is the arithmetic behind the ratios and differences
printed in the text, recomputed by `paper/derive_numbers.py` in the manuscript directory.
