# TENSORBOARD.md — Experiment Logging Contract

## Goal
Make every run inspectable, comparable, and reproducible via TensorBoard.
A run is considered incomplete if TensorBoard logs are missing.

## Canonical location
For every Stage 1 run directory:
<run_dir>/tensorboard/   (TensorBoard event files)

Where run_dir is:
artifacts/runs/{encoder_name}/{objective_name}/{run_id}/

## Mandatory logs (Stage 1)
### A) Resolved config (single source of truth)
Log the fully resolved config (same content as config.json):
- tag: config/resolved_json (text)

Also log the key hyperparams via HParams plugin:
- encoder.name
- objective.name
- window_timesteps (w)
- fs
- hop_train_timesteps
- hop_infer_timesteps
- embedding_dim
- clip_sigma (if used)
- seed
- optimizer, lr, batch_size, epochs

### B) Scalars (per step and/or per epoch)
Required:
- loss/train
- loss/val
- lr

Optional but recommended:
- grad_norm
- weight_norm
- any objective-specific diagnostics (e.g., contrastive temperature stats)

### C) Run identity (text)
- run/run_id
- run/timestamp

## Optional logs (Stage 2)
Stage 2 may also log to:
artifacts/stage2/{run_tag}/tensorboard/

Recommended scalars:
- diagnostics/pca_explained_variance_k
- diagnostics/inertia_vs_k
- diagnostics/silhouette_vs_k
- cluster/num_clusters
- smooth/window_size

## Determinism + comparability rules
1) Use consistent tag names across runs.
2) Prefer epoch-level logging for comparability; step-level is allowed.
3) Do not log raw sensor arrays; log summaries only.
4) Log the resolved config BEFORE training starts (so crashes still record config).

## How to run TensorBoard
tensorboard --logdir artifacts --port 6006
