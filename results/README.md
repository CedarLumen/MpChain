# Raw reports

Every file here was written by one of the scripts in `src/`, on the machine named inside it. None
was edited by hand, and none was retyped. That is the point of keeping them: the numbers printed in
the manuscript can be traced back to a file that a program produced.

## Where each file comes from

| File(s) | Written by | What it holds |
|---|---|---|
| `EVAL_UN.txt`, `EVAL_OPUS.txt`, `EVAL_ABLATION.txt`, `ABLATE_EXTRA.txt` | `eval_bleu_any.py` | corpus BLEU-2, precision at 1 and 2, token accuracy per checkpoint |
| `STATS.txt` | `stats_seeds.py` | per-seed values, paired differences, bootstrap intervals, sign counts |
| `BENCH.txt`, `BENCH_LADDER*.txt`, `BENCH_VARIANTS.txt` | `bench_infer2.py` | wall-clock along the optimisation ladder |
| `BENCH_BOTH.txt`, `bench_both_local.txt` | `bench_both.py` | both architectures timed at the same optimisation level |
| `EDGE_LOCAL.txt` | `bench_edge.py` | laptop GPU and CPU: latency, board power, energy per token |
| `PI5_2GB_summary.txt`, `PI5_2GB_raw.txt` | `pi_bench.py` on the board | Raspberry Pi 5, 2 GB, CPU only: latency and peak resident memory |
| `EXTEND_T.txt` | `eval_extend_t.py` | accuracy of one set of weights re-instantiated with a longer chain |
| `flops.txt` | `flops_report.py` | FLOPs per token under the counting convention of the paper |
| `route_sens.txt`, `route_shuffle*.txt`, `route_v2.txt` | `route_sensitivity.py`, `route_shuffle.py` | routing perturbation and the permutation control |
| `derived.txt`, `derived2.txt` | `make_derived_report.py`, and the manuscript's `derive_numbers.py` | the arithmetic behind every ratio and difference quoted in the text |
| `ALL14.txt`, `agg.txt`, `EXTRA2.txt`, `T3*.txt`, `t3_*.txt`, `tf*.txt` | intermediate runs of the same scripts | ablations, step caps and the dense baseline at earlier checkpoints |
| `diag_*.txt` | corpus diagnostics | vocabulary coverage, conditional entropy per length bucket, encoding checks |

## A note on the column labels

These logs were produced by the builds that ran the experiments. At that time the scripts wrote
their column labels in Chinese; the scripts released here write the same columns in English, and
re-running any of the commands in `docs/reproduction.md` reproduces the same numbers with English
headers. The logs were kept verbatim rather than retyped, so a few labels need a glossary:

| Label in these files | Meaning |
|---|---|
| 本方法 | this work (the chain) |
| 稠密 / 稠密基线 | dense Transformer baseline |
| 标准 MoE / 标准 MoE 基线 | token-level mixture-of-experts baseline |
| 主模型 | main model |
| 均值 / 标准差 / 平均 | mean / standard deviation / average |
| 平均步数 | mean number of calls per prediction |
| 汇总 | summary block |
| 核验 | sanity check (decoded tokens, sanity print) |
| 对照 | control arm |
| 总参 / 权重MB / 峰值MB / 总显存 | total parameters / weights in MB / peak memory in MB / total device memory |
| 首token ms / 每token / 吞吐 tok | first-token milliseconds / per-token milliseconds / throughput in tokens per second |
| 算得 / 正文 | computed value / value printed in the manuscript (columns of `derived2.txt`) |
| 逐 seed / 逐句记录 | per seed / per-sentence records |
| 去掉公共存储 | shared memory removed |
| 随机路由 | routing replaced by a fixed permutation |
| 只监督最后一步 | supervision on the last call only |
| 自适应早停 | learned termination |
| 数据集 / 评估 / 原始 | dataset / evaluation / original |
| 线程 / 单位 KB | thread count / unit is kilobytes |
| 表示本方法更慢 | a ratio above one means the chain is slower |
