# CONFIGS.md — Experiment Configuration Schema (Single Source of Truth)

This file defines the canonical config structure for Stage 1 (encoder learning + embedding extraction)
and Stage 2 (diagnostics + clustering + smoothing + segmentation).

Principle: **experiments are config-driven**. Code should not hardcode experiment choices.

---

## 1) Core rules (non-negotiable)

1) **Registry selection**
- Components are selected by name via registries:
  - `encoder.name`
  - `objective.name`
  - `clusterer.name` (Stage 2)
  - `smoother.name` (optional Stage 2)

2) **All objective-specific hyperparameters live in one place**
- ✅ `objective.kwargs.*`
- ❌ No separate namespaces like `contrastive.*`, `predictive.*`, `masked.*`.

3) **Resolved config is the truth**
- Every run must write `config.json` which is the fully resolved config (defaults applied).
- Reproducing a run should be possible from `config.json` alone.

4) **Filesystem roots are configurable**
- CLI must accept `--artifacts_root` and `--data_root` (defaults are `artifacts/` and `data/`).
- Path generation must use `traceability/utils/paths.py` (see docs/PATHS.md).

---

## 2) Stage 1 config schema (YAML)

### Required top-level keys
```yaml
run:
  seed: 123
  run_id: null            # optional; if null, code generates timestamp+hash

io:
  artifacts_root: artifacts
  data_root: data

data:
  fs: 3
  window_timesteps: 18
  hop_train_timesteps: 3
  hop_infer_timesteps: 1
  channels: [accel_pedal_pct, steer_angle_deg, steer_speed, speed, brake_kpa, yaw_deg_s, lturn, rturn]
  normalization:
    zscore: true
    clip_sigma: null      # e.g. 5.0; null disables clipping
    binary_channels: [lturn, rturn]   # never zscore these

encoder:
  name: tcn_ae
  kwargs: {}

objective:
  name: ae_mse
  kwargs: {}

training:
  epochs: 50
  batch_size: 128
  optimizer:
    name: adamw
    kwargs:
      lr: 3.0e-4
      weight_decay: 1.0e-2
  scheduler:
    name: null
    kwargs: {}
  grad_clip_norm: null
  device: auto
  num_workers: 0
  log_every_steps: 50
  eval_every_epochs: 1

logging:
  tensorboard: true
  tb_flush_secs: 10
  save_hparams: true



Objective kwargs - examples:
1. Autoencoder reconstruction (AE / MSE)
objective:
  name: ae_mse
  kwargs:
    recon_loss: mse   

2. Contrastive (InfoNCE-style)
objective:
  name: info_nce
  kwargs:
    temperature: 0.1
    augmentations:
      - name: jitter
        kwargs: {sigma: 0.02}
      - name: scale
        kwargs: {min: 0.9, max: 1.1}
    aug_seed: 123                 # if null, default to run.seed
    projection_head:              # optional; must NOT redefine encoder.encode output
      enabled: true
      dims: [128, 64]

3. Predictive (future window prediction)
objective:
  name: predictive_future
  kwargs:
    future_offset_timesteps: 3    # >=1; predicts window ending at t_end + offset

4. Masked modeling
objective:
  name: masked_recon
  kwargs:
    mask_prob: 0.15
    mask_mode: time              # "time" | "channel" | "time+channel"
    mask_value: 0.0              # or "learned_token" if encoder supports it
    mask_binary_channels: false



Complete Stage 1 YAMLs - examples: 

Example A — TCN AE + reconstruction
run: {seed: 123, run_id: null}

io: {artifacts_root: artifacts, data_root: data}

data:
  fs: 3
  window_timesteps: 18
  hop_train_timesteps: 3
  hop_infer_timesteps: 1
  channels: [accel_pedal_pct, steer_angle_deg, steer_speed, speed, brake_kpa, yaw_deg_s, lturn, rturn]
  normalization:
    zscore: true
    clip_sigma: 5.0
    binary_channels: [lturn, rturn]

encoder:
  name: tcn_ae
  kwargs:
    latent_dim: 20
    hidden_channels: 64

objective:
  name: ae_mse
  kwargs:
    recon_loss: mse

training:
  epochs: 50
  batch_size: 128
  optimizer: {name: adamw, kwargs: {lr: 3.0e-4, weight_decay: 1.0e-2}}
  scheduler: {name: null, kwargs: {}}
  grad_clip_norm: 1.0
  device: auto
  num_workers: 0
  log_every_steps: 50
  eval_every_epochs: 1

logging: {tensorboard: true, tb_flush_secs: 10, save_hparams: true}

Example B — TCN encoder + InfoNCE
run: {seed: 123, run_id: null}
io: {artifacts_root: artifacts, data_root: data}

data:
  fs: 3
  window_timesteps: 18
  hop_train_timesteps: 3
  hop_infer_timesteps: 1
  channels: [accel_pedal_pct, steer_angle_deg, steer_speed, speed, brake_kpa, yaw_deg_s, lturn, rturn]
  normalization:
    zscore: true
    clip_sigma: null
    binary_channels: [lturn, rturn]

encoder:
  name: tcn_encoder
  kwargs:
    embedding_dim: 64
    hidden_channels: 64

objective:
  name: info_nce
  kwargs:
    temperature: 0.1
    aug_seed: null
    augmentations:
      - {name: jitter, kwargs: {sigma: 0.02}}
      - {name: scale,  kwargs: {min: 0.9, max: 1.1}}
    projection_head: {enabled: true, dims: [128, 64]}

training:
  epochs: 50
  batch_size: 256
  optimizer: {name: adamw, kwargs: {lr: 3.0e-4, weight_decay: 1.0e-2}}
  scheduler: {name: null, kwargs: {}}
  grad_clip_norm: null
  device: auto
  num_workers: 0
  log_every_steps: 50
  eval_every_epochs: 1

logging: {tensorboard: true, tb_flush_secs: 10, save_hparams: true}


