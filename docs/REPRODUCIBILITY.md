# Reproduction guide

## 1. Artifacts and protocols

| Artifact | Role | Fixed information |
|---|---|---|
| `checkpoints/es_ngs_dm.pt` | Paper online deployment model | Original tensor values; metadata paths removed |
| `results/paper_*_es_ngs_dm.csv` | Paper ES-NGS-DM runs | 30 Synthetic and 8 LaDe cases; five search seeds |
| `data/synthetic/` | Frozen synthetic geometry collection | 900 geometries, 100/200/300/500/1,000 customers, six spatial configurations |
| `data/splits/paper_evaluation.csv` | Exact paper evaluation list | 30 n=1,000 QP/RP/LP cases and 8 LaDe cases |
| `data/training/` | 2026-11-08 supplementary training pack | 480 states, 40 instances; 120 base plus 360 replay states |
| `data/splits/supplement/` | Supplementary held-out validation | Five instance-grouped folds, all states from an instance in one fold |

Training geometries use seeds 20269901–20269924 (n=200) and
20269941–20269956 (n=1,000), cycling through QP/RP/LP. These seeds are separate
from the frozen spatial benchmark seeds 20261011–20261040.
The exact paper subset is selected by the saved run records, not by reranking
the available geometries. Some paper cases retain the original generator's
`development` metadata: that original designation is preserved in the geometry
manifest rather than relabeled as a new holdout split.

The supplementary states are newly collected search states, not the historical
states used to obtain `deployment_all.pt`. Fixed geometry and random seeds do
not fix wall-clock-limited parent-search trajectories. The historical state
pack, its auxiliary utility table and its complete Python environment lock
were not recovered into this distribution. The final historical deployment
weights and their recorded arguments are included. Use the released model for
paper online evaluation and the supplementary pack for a complete new training
run. Do not report supplementary training metrics as the historical independent
five-fold results.

## 2. Model, labels and objective

The online network takes `X[C,32,24]`, `A[C,33,33]`, `action[C,32]`,
`global[C,16]` and `mask[C,33]`. Node tokens and the CLS token are encoded by
three attention layers (width 64, four heads, FFN width 256, dropout 0.1).
Relation embeddings are mean pooled and added to tokens. Action and global
features are projected into the decision context. Five heads predict total
gain, driving contribution, swapping contribution, synchronization contribution
and repairability. The total trainable parameter count is 205,253.

The objective and label are:

```text
minimize T(solution), the delivery completion time in hours
y = 100 * (T_before - T_after_exact_repair) / T_before
score = predicted_gain + 1.0 * log(max(sigmoid(feasibility_logit), 1e-6))
```

`T_after_exact_repair` is used for supervision, never as an online feature.
Features use the current feasible solution and raw candidate structure.
The exact labels contain driving, swapping and synchronization contributions
on the same percentage-gain scale. Failed repairs receive infeasible labels.

`model.py` implements gain MSE plus 0.5 times each component MSE, and a 0.1
consistency MSE. Training also uses listwise exact-label ranking, BCE
repairability, exact soft targets and optional teacher preferences. See
`configs/training.json` for every outer loss weight, optimizer, schedule and
seed. No weighted energy-cost objective is used by this release.

## 3. Online search

Install the external initialization dependencies before running Julia:

```bash
python scripts/setup_solver_dependencies.py
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

The Julia package is `ESNGSDM`. Upstream solver sources are installed outside
the repository, with pinned revisions and hashes. See
[dependency setup](DEPENDENCIES.md) for custom paths and the runtime boundary.

```bash
julia --project=. scripts/run_case.jl --benchmark synthetic --case n1000_k12_seed20261014_QP --seed 1 --budget 60 --python python
julia --project=. scripts/run_case.jl --benchmark lade --case HZ_courier15_ds520_aoi22 --seed 1 --budget 60 --python python
```

Use `--python` with the actual environment interpreter when necessary.
Paths with spaces are supported by tab-delimited scorer communication.
Temporary score directories are exclusive to each call and removed after use.
The runner saves uniquely named result files and never overwrites the frozen
case geometry or model checkpoint.

Physical settings are 40/60 km/h truck/drone speeds, 100/20 km-equivalent
capacities, shared-energy factor 0.4 and two-minute full truck-battery swapping.
Launch/retrieval service times use the instance builder's recorded defaults
of 100/20 seconds (launch service is zero at the starting depot). Truck movement uses the instance's Manhattan metric; drone flight
uses Euclidean distance. The Pareto station-path repair handles replenishment
and subsequent exact evaluation checks synchronization and energy feasibility.

Each round samples a window of at most three synchronization intervals and
up to 64 unique candidates. Top-1 is completely repaired. The RNG seed is the
reported search seed plus 300001. With less than two seconds remaining the
candidate pool limit is 32; the search has a 0.35-second iteration guard and
a five-second outer reserve. Only a feasible improvement of more than 1e-9 h
is accepted. `configs/deployment.json` records these settings.

For a deterministic work-limited smoke rather than a wall-clock experiment:

```bash
julia --project=. scripts/run_case.jl --benchmark lade --case HZ_courier15_ds520_aoi22 --seed 1 --budget 120 --max-rounds 3 --python python
```

The result is labeled `fixed_work`. It is not a 60-second benchmark run.
`--max-exact 128` instead imposes a complete-repair call cap.

Charged online time is `init_s + preprocess_s + search_s + validate_s` in the
paper aggregation. The public fresh-run entry point follows this accounting
and records these components separately. Python model loading is a separately reported
deployment startup cost. Compilation, package loading and reading geometry
precede the timed initialization. Run all methods/seeds serially on the same
hardware for timing comparisons; equal seeds under different CPU contention
do not imply identical wall-clock search trajectories.

## 4. Student training

The portable wrapper reads the supplied state index and fixed grouped folds.
The all-data index is for deployment training; its training diagnostic is not
held-out accuracy.

```bash
python scripts/train.py --stage base
python scripts/train.py --stage exact
python scripts/train.py --stage qwen --teacher-utilities data/training/teacher_utilities_supplement.csv
```

`base` uses 40 epochs, then matched `exact` and `qwen` branches each use
12 additional epochs from the same base. Fixed final checkpoints are saved.
Outputs go to `outputs/training/deployment/`; they never replace the paper
checkpoint. The included auxiliary utilities are generated by the all-data
supplementary teacher and are suitable for all-data deployment training.

For held-out fold 1, run a fold-specific teacher and utility export:

```bash
python scripts/train.py --stage base --fold 1
python scripts/train.py --stage exact --fold 1
python scripts/train.py --stage teacher --fold 1
python scripts/train.py --stage export --fold 1
python scripts/train.py --stage qwen --fold 1
```

Repeat for folds 2–5. Every fold's teacher is trained from that fold's training
instances only; validation metrics are logged but do not select the checkpoint
or training length. Never reuse the all-data teacher as a held-out-fold teacher.

The historical independent exact-only protocol used a different 20-epoch,
random-initialized schedule with seed `20261027 + fold`. Its recorded
configuration is in `configs/training.json`. To run the same schedule on the
released supplementary fold 1 (a new dataset-specific validation), use:

```bash
python repro/es_ngs/qwen_exp/scripts/train_student_fixed_epoch.py --variant full_v3 --es-ngs-root repro/es_ngs --label-dir data/training --state-index data/splits/supplement/fold1.csv --soft-csv data/training/exact_soft_labels.csv --out-ckpt outputs/exact_cv/fold1.pt --epochs 20 --batch-size 4 --lr 0.0002 --rank-w 1 --soft-w 0.5 --kd-w 0 --feas-w 0.5 --online-feas-weight 1 --seed 20261028
```

## 5. Qwen decision model and preference export

```bash
python -m pip install -r requirements-qwen.txt
python scripts/download_qwen.py
python scripts/train.py --stage teacher
python scripts/train.py --stage export
python scripts/train.py --stage qwen
```

The download is pinned to the commit in `configs/qwen_decision.json`.
Qwen decision-head training requires CUDA. The backbone uses the first eight
pretrained layers with LoRA rank 8 on `q_proj` and `v_proj`, alpha 16 and dropout
0.05. Candidate-position hidden states and a numeric residual feed choice,
gain, driving, swapping, synchronization and feasibility heads. The teacher
learns a residual correction to a frozen structural prior.

After running the `teacher` stage, export preferences from that trained model:

```bash
python repro/es_ngs/qwen_exp/scripts/export_jev_decision_utilities.py --manifest outputs/training/deployment/teacher/jev_qwen_math.pt --label-dir data/training --state-index data/training/state_index_all.csv --esngs-root repro/es_ngs --output outputs/supplement_utilities.csv
```

The local base model is `models/Qwen2.5-1.5B`; `ESNGS_QWEN_BASE` can select
another local copy of the same pinned revision. The exported utility is the
structural score plus the learned Qwen residual correction. The included
preference CSV was exported with residual scale 0.10665960609912872.
These utilities supervise the lightweight online student. Qwen adapters and
later supplementary student weights are not distributed; the only bundled
weight is the Synthetic-trained final paper deployment network.

The older text-ranking route is retained separately:

```bash
python repro/es_ngs/qwen_exp/scripts/build_decision_dataset.py --label-dir data/training --out-dir outputs/text_decisions --seed 42 --val-frac 0.2
python repro/es_ngs/qwen_exp/scripts/train_qwen_lora.py --model-dir models/Qwen2.5-1.5B-Instruct --data-dir outputs/text_decisions --out-dir outputs/qwen_sft --epochs 3 --lr 0.0002 --batch-size 1 --grad-accum 8 --max-seq-len 8192 --lora-r 16 --lora-alpha 32 --seed 0
```

This optional route requires the corresponding official Instruct model,
downloaded separately. It masks prompt tokens and computes response-only SFT.
It is not the direct non-generative decision-head protocol used to create the
supplementary preferences. The overlap-group and Bradley–Terry utilities are
included for that historical ranking route; they are not called online.

## 6. Data generation

The supplied `.jls` files preserve the frozen coordinates and geometry hashes.
Use Julia 1.10.5 to read them. New generation writes separate output directories.

```bash
julia --project=. scripts/freeze_structured_v3.jl
```

This generates the full 900-instance collection into `data/synthetic_generated/`
and refuses to overwrite an existing manifest. `ESNGS_GEOMETRY_OUT` can select
a different new output directory. Demand profiles Q/R/L are four corner
clusters, a peripheral annulus, and two long corridors; station profiles C/P
are central/peripheral. The paper evaluation uses only the exact 30 cases in
`paper_evaluation.csv`, not every generated case.

To regenerate LaDe scenarios from the official Hangzhou parquet:

```bash
python -m pip install -r requirements-data.txt
python scripts/build_lade_hz_cases.py --parquet /path/to/delivery_hz-00000-of-00001-8090c86f64781f71.parquet
julia --project=. scripts/freeze_lade_hz_cases.jl
```

The required input SHA256 is
`00b6b4dca471cadee796a3fc7f07dcd919499ec58459ea67db784dcb06089e75`.
Coordinates are AOI median locations projected into local km and centered at
their mean. The three courier-day and five region-day selection rules are in
the generator. Candidate stations are synthetic, seeded from scenario identity.
Raw LaDe parcel records and base Qwen weights are downloaded from their owners,
not redistributed in this repository.

For fresh exact-label collection, set `ESNGS_LABEL_OUT` to distinct new
directories and run these in order:

```bash
julia --project=. repro/es_ngs/collect_labels_scale.jl
julia --project=. repro/es_ngs/collect_labels_replay.jl
python repro/es_ngs/qwen_exp/scripts/merge_label_packs.py --base repro/es_ngs/experiments/labels_v2 --replay repro/es_ngs/experiments/labels_v2_replay --out outputs/labels_merged
```

Default output directories are separate from the released training pack.
The base collector uses 48 candidates/state and three phases/instance; replay
uses 64 candidates/state and nine phases/instance. Parent trajectories use
wall-clock advances. For exact repeatability of supervised learning inputs,
use the frozen released states rather than recollecting them.

## 7. Integrity and validation

```bash
python scripts/verify_release.py
python scripts/check_dependency_boundary.py
python -m unittest discover -s tests -v
python repro/es_ngs/qwen_exp/scripts/test_overlap_bt.py
python repro/es_ngs/qwen_exp/scripts/smoke_score_isolation.py --ckpt checkpoints/es_ngs_dm.pt --score-server repro/es_ngs/score_server.py --workers 4 --workdir outputs/isolation
```

The hash manifest covers code, data, weights, configuration and figures.
Group checks confirm disjoint training/validation instances in every fold.
The CPU tests load the released weights, score real candidate features, check
finite forward/backward results and inspect the Qwen heads. The isolation
smoke checks identical scores across four independent processes.

The release does not start any training or benchmark jobs automatically.
