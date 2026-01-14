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

