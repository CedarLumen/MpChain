# Dissecting Serial Sparse Activation

Code, raw measurement logs and derived quantities for the manuscript

> **Dissecting Serial Sparse Activation: On the Necessity of Chain Depth and What Memory and Routing Each Carry**
> Mengzhe Chen, Beijing Institute of Technology

The paper asks what a chain of hard-selected experts carries. This repository contains the
instrument used to answer that question, the controls that interrogate it, and the logs the
reported numbers are read from.

## What is in the repository

| Path | Contents |
|---|---|
| `src/` | Model, training loops, evaluation, mechanism ablations, efficiency benchmarks (20 files) |
| `scripts/` | End-to-end reproduction, one script per block of results |
| `results/` | The raw reports the paper's numbers are taken from (machine-written text files) |
| `docs/` | `reproduction.md` (paper section to command to output) and `data.md` (corpora) |
| `paper/` | Manuscript and cover letter |

`src/` files:

| File | Role |
|---|---|
| `mpchain_tr.py` | The chain: per-expert encoder, fixed-length shared memory, hard argmax routing, termination head |
| `train_tr.py` | Training loop for the chain (per-step supervision, routing-collapse suppression, clones) |
| `train_moe.py` | Token-level mixture-of-experts baseline at matched parameters |
| `train_tf.py` | Dense causal Transformer baseline at matched parameter count |
| `data.py`, `data_opus.py` | Corpus to character-id conversion (United Nations corpus, OPUS-100) |
| `eval_bleu_any.py` | Greedy decoding, corpus BLEU-2, per-sentence records for paired statistics |
| `eval_tf_bleu.py` | The same evaluation path for the dense baseline |
| `stats_seeds.py` | Cross-seed means, paired differences, bootstrap intervals |
| `slot_ablation.py` | Which memory slots carry the sentence-level metric |
| `route_shuffle.py` | Permutation control: expert-usage marginal preserved, content correspondence broken |
| `route_sensitivity.py` | Character-level perturbation of the routing decision |
| `eval_extend_t.py` | Re-instantiating trained weights with a longer chain |
| `flops_report.py` | FLOPs per token under a fixed counting convention |
| `bench_infer2.py` | Wall-clock on an optimisation ladder (eager, static buffers, compiled, CUDA graph) |
| `bench_both.py` | Both architectures timed at the same optimisation level |
| `bench_edge.py` | Laptop GPU and laptop CPU: latency, board power, energy |
| `pi_bench.py`, `make_pi_data.py` | Raspberry Pi 5 (2 GB, CPU only): latency and peak resident memory |

## Install

Python 3.10 or newer.

```
pip install -r requirements.txt
```

CUDA is used when available; the same code runs on CPU, which is how the single-board measurements
were taken. `torch.cuda` calls are guarded by availability except in `bench_infer2.py` and
`bench_both.py`, which are GPU-only by construction.

## Data

Two public corpora, both converted once into character-id archives. See `docs/data.md`.

```
export MPCHAIN_SRC=/path/to/un_pc_en-zh_0.parquet
export MPCHAIN_OUT=$PWD/data/un_sub.npz
PYTHONPATH=src python src/data.py
```

The scripts expect the archive at `data/un_sub.npz` relative to the working directory, so run
everything from the repository root. `MPCHAIN_DATA` overrides the path for the evaluation scripts.

## Reproduce

`docs/reproduction.md` maps each block of results to one command and one output file. The short
version, on a machine with one CUDA device:

```
bash scripts/run_all.sh        # data, three training runs, evaluation, statistics
bash scripts/05_ablations.sh   # mechanism ablations, after training
bash scripts/06_efficiency.sh  # FLOPs and wall-clock
```

`scripts/run_all.sh` trains three models of 100k steps each. The wall-clock of every training run is
in the corresponding log under `results/`.

## Hardware

| Measurement | Device |
|---|---|
| Main results, ablations, FLOPs, optimisation ladder | RTX PRO 6000, 95 GB |
| Second device class | RTX 5060 Laptop, 8 GB, and the CPU of the same machine |
| Third device class | Raspberry Pi 5, 2 GB, four ARM A76 cores, CPU only |

## What is not in the repository

Checkpoints are about 4 GB and are not tracked here; they are available from the author on request.
The conversion of the United Nations release into the parquet layout the scripts read is not
included, because the upstream release layout changes between versions; `docs/data.md` gives the
schema and the filtering rule, and any converter producing that schema works.

## Citation

See `CITATION.cff`, or

```
@article{chen2026dissecting,
  title  = {Dissecting Serial Sparse Activation: On the Necessity of Chain Depth and What
            Memory and Routing Each Carry},
  author = {Chen, Mengzhe},
  year   = {2026},
  note   = {Manuscript submitted to Neurocomputing}
}
```

## License

MIT, see `LICENSE`.

## Contact

Mengzhe Chen, Beijing Institute of Technology, Beijing 100081, China.
cedargrove@foxmail.com
