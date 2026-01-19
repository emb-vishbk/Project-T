# PATHS.md — Path Conventions (Single Source of Truth)

## Goal
Prevent path drift and hardcoded file locations across the codebase.
All filesystem paths must be generated via traceability/utils/paths.py.

## Canonical layout

artifacts/
  runs/
    {encoder_name}/
      {objective_name}/
        {run_id}/
          weights_last.pt
          weights_best.pt
          config.json
          metrics.json
          tb/                 # TensorBoard event logs
  embeddings/
    {split}/
      {session_id}.npy
      {session_id}_t_end.npy

## Contract
1) No module may hardcode strings like "artifacts/..." except inside paths.py.
2) Stage scripts (CLI) must accept configurable roots:
   - --artifacts_root (default: artifacts)
   - --data_root (default: data)
3) paths.py must be pure: no IO on import; no side effects; no heavy deps.

## Required paths.py API (minimum)
- get_artifacts_root(artifacts_root: str | Path) -> Path
- run_dir(artifacts_root, encoder_name, objective_name, run_id) -> Path
- tb_dir(run_dir) -> Path
- embeddings_dir(artifacts_root, split) -> Path
- embeddings_file(artifacts_root, split, session_id) -> Path
- t_end_file(artifacts_root, split, session_id) -> Path

## Notes
- encoder_name and objective_name must be filesystem-safe registry IDs.
- run_id must be unique; recommended: timestamp + short hash of resolved config.


### High level Project Structure:

Project-T/
├─ AGENTS.md
├─ requirements.txt
├─ docs/
│  ├─ CONFIGS.md
│  ├─ PATHS.md
│  └─ TENSORBOARD.md
│
├─ configs/                 # new: experiment configs (YAML)
│  ├─ stage1/
│  └─ stage2/
│
├─ traceability/            # new: actual python package
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ paths.py           # "one-stop path oracle"
│  │  ├─ seed.py
│  │  └─ logging.py
│  ├─ config/
│  │  ├─ load.py            # YAML -> resolved config
│  │  └─ schema.py          # validation, defaults, resolved_config.json
│  ├─ data/
│  │  ├─ hdd_io.py
│  │  ├─ normalization.py
│  │  ├─ indexing.py
│  │  └─ dataset.py
│  ├─ models/
│  │  ├─ encoders/
│  │  └─ components/
│  ├─ objectives/
│  ├─ stage1/
│  │  ├─ train.py
│  │  └─ extract_embeddings.py
│  └─ stage2/
│     ├─ clusterers/
│     ├─ smoothers/
│     ├─ segment.py
│     └─ run.py
│
├─ cli/                     # new: thin entrypoints only (argparse)
│  ├─ train_stage1.py
│  ├─ extract_embeddings.py
│  └─ run_stage2.py
│
├─ notebooks/               # new: exploration, not core logic
├─ tests/                   # new: tiny but sharp
│
├─ data/                    # keep EXACTLY as you have
│  ├─ index/
│  ├─ meta/
│  └─ raw/
│
└─ artifacts/               # new: everything generated (gitignored)
   ├─ runs/                 # weights, resolved config, metrics, tensorboard
   ├─ embeddings/           # per split/session latents + t_end arrays
   └─ stage2/               # pca, kmeans, segments, reports
