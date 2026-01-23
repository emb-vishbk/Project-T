Project context:
We are building an end-to-end unsupervised/self-supervised pipeline for ego-driving event/scenario discovery using the Honda Driving Dataset (HDD), starting with the easy 3 Hz synchronized subset. The backbone pipeline follows Kreutz et al. (arXiv:2301.04988): sliding windows over multivariate time series, an encoder that maps each window to a latent embedding, clustering in latent space, and segmentation via cluster-ID changepoints. Labels are available, but must be used only for evaluation and sanity checks, not for training.

Current workspace layout (exact)
- .venv/                          python environment (ignore in code)
- 20200710_sensors/sensor/        137 session files, each <session_id>.npy, shape (N, 8)
- 20200710_labels/target/         matching label files, each <session_id>.npy, shape (N,)
- AGENTS.md                       this file
- dataset_creation.ipynb          notebook file for preparing the dataset

Data details
Each session is a variable-length sequence sampled at 3 Hz. The row index is time (uniform sampling). There is no unix_timestamp in the .npy files, and we do not add one for this 3 Hz subset. For any time-based computations, use dt = 1/3 seconds.

Sensor channels (X has shape N x 8, fixed column order)
0 accel_pedal_pct
1 steer_angle_deg
2 steer_speed
3 speed
4 brake_kpa
5 lturn (binary 0/1)
6 rturn (binary 0/1)
7 yaw_deg_s

Labels (y has shape N)
- Integer scenario IDs. Globally there are 12 labels in the dataset.
- Labels are aligned to the same 3 Hz timeline as sensors.

Core method to implement (paper-aligned)
We treat each window as a subsequence ending at an index t_end:
- window length w (timesteps)
- X_window = X_session[t_end-(w-1) : t_end+1]  shape (w, 8)
We attach the embedding z_t to the end time t_end (timestamp-level representation).
Then per session we create an embedding sequence Z_session in time order, cluster embeddings to obtain a discrete state sequence, and segment by changepoints where cluster id changes.

Windowing configuration
Sampling rate fs = 3 Hz
Window length is defined in discrete timesteps:
- window_timesteps w (integer) is the primary config hyperparameter.
- The equivalent duration is L_seconds = w / fs (for logging only).
Two hop values are configurable hyperparameters (in timesteps):
- hop_train_timesteps for encoder training
- hop_infer_timesteps for inference / clustering
Typical defaults (for 3 Hz HDD) might be, for example, hop_train_timesteps = 3 and hop_infer_timesteps = 1, but these must come from config, not be hardcoded.
Rationale: training can use a larger hop to reduce redundancy, while inference can use a smaller hop for finer segmentation, but both are just configuration choices.

Unified dataset design (do not duplicate data)
Do not pre-slice and save all windows to disk.
Instead, build an index that points into the existing session arrays.

Index entry definition
Minimal index entry:
- session_id, t_end
Derived start index:
- t_start = t_end - (w - 1)

We maintain two index families (training vs inference), and each family exists per split ⇒ up to 6 index files: {training,inference} × {train,val,test}
train index: built with hop_train_timesteps from config
infer index: built with hop_infer_timesteps from config
Both indices should be reproducible with fixed parameters and seed. They are just window indices constructed with different hop values..

Splitting strategy (critical)
Split by session_id, never by windows, to avoid leakage due to overlapping windows from the same drive.
Recommended: 70/15/15 train/val/test split by session IDs with a fixed random seed.
After splitting sessions, build window indices separately for each split.

Normalization and preprocessing
Keep preprocessing minimal initially.
Compute per-channel mean and std on train split only (across all timesteps in train sessions).
Apply the same normalization to val/test.
Binary channels lturn/rturn must remain 0/1 (do not z-normalize them).
Compute mean/std only for continuous channels and apply z-score + optional clipping only to those.
For lturn/rturn, treat them as pass-through features (but values stay exactly 0/1).

Post z-score clipping (optional but recommended): allow the dataset to apply a configurable clipping after z-score normalization on continuous channels.
Expose a clip_sigma hyperparameter (e.g., default 6.0) in config instead of hardcoding it.
Clipping is applied as z = clip(z, -clip_sigma, +clip_sigma) only on continuous channels; binary channels (lturn, rturn) are left unchanged.
This should be treated as a preprocessing knob that can be turned on/off or adjusted during experiments, not as a fixed design choice.

What the dataset must support (future-proof)
Training stage (encoder learning)
- Base dataset returns a single view:
  {"x": window, "meta": {"session_id": ..., "t_end": ...}}
- Objective-specific view/batch builder (wrapper dataset or collate_fn)
  takes this base sample and constructs any additional keys listed in
  REQUIRED_BATCH_KEYS (e.g. x1, x2, x_future, mask, target, …).
- The base dataset must not bake in objective-specific logic; it only
  handles indexing, loading, normalization, and clipping.

Inference stage (embedding extraction for clustering)
- Iterate windows in time order per session using infer index
- Produce embedding sequence Z_session aligned by t_end
- Cluster embeddings to get cluster_id[t_end]
- Segment by changepoints in the cluster id sequence

Label alignment for evaluation
Simplest alignment: assign window label as y[t_end] (label at window end).
Alternative: majority label over the window or center label; keep this configurable.

Deliverables expected from the agent (implementation plan)
All data-prep outputs (dataset_summary, splits, normalization, indices) live under artifacts_root/data_info/{meta,index}/.
1) Dataset scan and validation
- Iterate all files in 20200710_sensors/sensor and 20200710_labels/target
- Verify each session_id exists in both folders
- Verify X.shape[0] == y.shape[0] per session
- Report dataset summary: number of sessions, lengths, duration in minutes, feature dtype, label inventory, per-label counts, per-session label diversity

2) Session split builder
- Build train/val/test lists of session_ids with a fixed seed
- Save splits to a small file, e.g. artifacts/splits.json

3) Window index builder
Input: session_ids, window length w (timesteps), and a hop value (timesteps).
Output: list of (session_id, t_end).
Save indices, e.g. artifacts/index_training_{train/val/test}.jsonl, for training; and artifacts/index_inference_{train/val/test}.jsonl if separate inference indices are needed.
Use hop_train_timesteps from config when building train/val/test indices for encoder training, and hop_infer_timesteps from config when building indices intended for inference/clustering.
eg. {"session_id":"","t_end":}

4) Dataset class
- Uses an index list and loads the needed session file, slices windows on demand
- Supports optional label return for evaluation
- Applies normalization consistently
- (recommended optimization): consider np.load with mmap_mode="r" and caching recently used sessions
- Expose a clip_sigma configuration option (e.g., via normalization.json or a model/config file) so post z-score clipping can be turned off (None), set to a default (e.g. 6.0), or swept in ablations without changing code.

5) Quick sanity checks
- Randomly sample a few windows, verify shape (w, 8) where w = window_timesteps from config
- Verify time alignment: label slicing matches sensor slicing lengths
- Plot a few channels for a random session to confirm signals look sane

Practical quickstart (what to run first)
- Run the dataset scan to confirm shapes and matching session IDs
- Create session splits
- Build train index using the configured window length w and hop_train_timesteps from config
- (Optionally) build a separate infer index using the same w and hop_infer_timesteps from config, if you want precomputed indices for inference/clustering
- Train an encoder using a config:
  python train_encoder.py --config configs/...
- Extract embeddings for all splits from a chosen run:
  python extract_embeddings.py --run_dir artifacts/stage1/{encoder_name}/{objective_name}/<run_id>/ --splits train,val,test
- Implement the dataset class and iterate a small batch to confirm correct window extraction and hop behavior

Notes and pitfalls
- Do not add timestamps to the sensor arrays for this 3 Hz subset; time is implicit.
- Do not shuffle windows across sessions for splitting; always split by session_id.
- Inference windows must be generated in time order per session for segmentation to make sense.
- Labels are evaluation-only; no supervised learning should be baked into the training objective in this phase.
- Treat post z-score clipping (clip_sigma) as a hyperparameter to ablate, especially when changing the encoder loss family (MSE, contrastive, predictive, masked modeling, etc.). For any new loss: start with the current “sane default” (e.g., clip_sigma = 6.0 on continuous channels).

North star outcomes for this phase
- Build a working pipeline: windows -> encoder embeddings -> clustering -> segments
- Show that clusters correspond to distinct driving regimes (e.g., braking vs accelerating vs turning vs cruising) via cluster prototypes and temporal coherence
- Use labels only to quantify sanity (purity/NMI) and to debug, not to train

Stage 1: Encoder learning (self-supervised/unsupervised)

Goal. Learn an encoder f_theta that maps each normalized window X(t_start:t_end) to an embedding z_t. Stage 2 only consumes per-session sequences of (z_t, t_end) and is agnostic to how f_theta was trained.

Config-driven model & objective.

* Encoder and training objective are selected via config (e.g. YAML), not hardcoded:

  * encoder.name, encoder.kwargs — e.g. tcn_ae, tcn_encoder, transformer_encoder, …
  * objective.name, objective.kwargs — e.g. ae_mse, info_nce, predictive_future, masked_mlm, …
  * Schema rule: ALL objective-specific hyperparameters MUST live under objective.kwargs. Do not create separate namespaces like predictive.*, contrastive.*, masked.* — those are examples of keys inside objective.kwargs.

* train_encoder.py is a generic trainer: it reads the config, instantiates encoder + objective from registries, and runs a single training loop.

Stable encoder interface (Stage 1 → Stage 2 contract).

* Every encoder must implement:

  * encode(x) -> z

    * x: window tensor after normalization/clipping. Must support batched input: shape (B, w, C). Single-window input (w, C) is allowed and treated as B=1.
    * z: embedding tensor of shape (B, embedding_dim) (or (embedding_dim,) for single-window).
* Additional methods (decode, reconstruct, etc.) are allowed for specific objectives, but encode is the public contract used by extract_embeddings.py and Stage 2.

Objective interface & batch requirements.

* Each objective declares the batch keys it requires, for example:

  * AE: REQUIRED_BATCH_KEYS = ["x"]
  * Contrastive: ["x1", "x2"]
  * Predictive: ["x", "x_future"]
  * Masked: ["x_masked", "mask", "target"]
* Each objective may also declare the encoder capabilities it requires, for example:

  * REQUIRED_ENCODER_METHODS = ["encode"]                 # most objectives
  * AE: ["encode", "decode"]                              # reconstruction-based objectives
  * Masked: ["encode"] (plus optional "mask_token" support if implemented)
* train_encoder.py must validate at startup that the selected encoder implements REQUIRED_ENCODER_METHODS for the selected objective, and fail fast with a clear error.

* Each objective implements a common API:
  * compute_loss(encoder, batch) -> (loss, logs_dict)
* The training loop never branches on objective/encoder type; it only calls compute_loss.

Dataset & batch construction.

* The base dataset (HDDWindowDataset) is simple and stable:

  * index entries: (session_id, t_end) with global window_timesteps (w) and fs from config
  * returns base samples:
    {"x": window, "meta": {"session_id": ..., "t_end": ...}}
* A lightweight view/batch builder (wrapper dataset or collate_fn) reads REQUIRED_BATCH_KEYS and constructs any extra fields needed (x1, x2, x_future, mask, …) from the base window(s), producing the final batch dict passed to objective.compute_loss.

* Batch Builder Spec (objective-driven; must be reproducible)
The batch builder (wrapper dataset or collate_fn) is the ONLY place where
objective-specific fields are created (x1/x2, x_future, masks, targets, ...).
It must be deterministic given (global_seed, session_id, t_end), so that
experiments are reproducible.

General rules:
- Never cross session boundaries when constructing additional views (same session_id only).
- Continuous channels may be augmented; binary channels (lturn, rturn) must remain 0/1.
- If a required view cannot be constructed due to boundary conditions, default behavior is:
  drop that sample (do not pad/clamp silently). Log counts.

A. Predictive objective: construct x_future
- Config (inside objective.kwargs):
  - objective.kwargs.future_offset_timesteps (integer >= 1)
- Definition:
  - Base window ends at t_end
  - Future window ends at t_end_f = t_end + future_offset_timesteps
  - x_future = X_session[t_end_f-(w-1) : t_end_f+1]
- Boundary:
  - If t_end_f > N-1 (future window exceeds session), drop sample.

B. Contrastive objective: construct x1, x2
- Config (inside objective.kwargs):
  - objective.kwargs.augmentations: list of augmentations + params
  - objective.kwargs.aug_seed: global seed (or reuse run seed)
- Definition:
  - x1 = augment(x), x2 = augment(x) (two independent draws)
  - Apply augmentations only on continuous channels; keep lturn/rturn unchanged.
- Reproducibility:
    - Seed each sample’s augmentation RNG using a STABLE hash of (aug_seed, session_id, t_end, view_id), e.g. via hashlib (sha1/md5) → 32-bit int seed. Do NOT use Python built-in hash(), since it may vary across runs/processes.

C. Masked modeling objective: construct x_masked, mask, target
- Config (inside objective.kwargs):
  - objective.kwargs.mask_prob (e.g. 0.15), objective.kwargs.mask_mode ("time", "channel", or "time+channel")
  - objective.kwargs.mask_value (e.g. 0.0) or "learned_token" (if model supports it)
- Definition:
  - mask is a boolean array with same shape as x (or same time length, depending on mode)
  - target contains the original values for masked positions only (or full x with loss masked)
  - x_masked is x with masked positions replaced by mask_value
- Channel rule:
  - Do not mask binary channels unless explicitly enabled (e.g., objective.kwargs.mask_binary_channels=true).

Train (using index built with hop_train_timesteps).
Assume artifacts/index_training_train.jsonl and artifacts/index_training_val.jsonl already exist for the configured window_timesteps (w) and hop_train_timesteps. A typical training run:

python train_encoder.py 
--config configs/ .....

The config must specify:

* data paths (sensor/label roots, train/val index paths)
* fs, window_timesteps, hop_train_timesteps, hop_infer_timesteps
* normalization/clipping (clip_sigma)
* encoder.* and objective.*
* optimizer/scheduler settings, seeds, device, logging/output dirs, etc.

Outputs in artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/ (per run).
Each run writes to a unique subdirectory under artifacts/stage1/{encoder_name}/{objective_name}/ (e.g. run_id) and must contain at least:


* weights_last.pt, weights_best.pt — encoder weights (sufficient to call encode)
* config.json — fully resolved config for the run (including encoder_name, encoder_kwargs, objective_name, objective_kwargs, fs, window_timesteps, hop_train_timesteps, hop_infer_timesteps, clip_sigma, seeds, paths)
* metrics.json — train/val curves and any objective-specific metrics logged over epochs

Extract embeddings (using hop_infer_timesteps for windowing).
After training, extract embeddings for all splits using the inference hop:

python extract_embeddings.py 
--run_dir artifacts/stage1/{encoder_name}/{objective_name}/<run_id>/ 
--splits train,val,test

extract_embeddings.py must:

* load the trained encoder (weights_best.pt + config.json) from the given run_dir
* build inference windowing using window_timesteps + hop_infer_timesteps from config (either by building a fresh infer index or using artifacts/index_inference_{split}.jsonl if available)
* call encoder.encode(x) in time order per session
* save, for each split and session:

  * artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}.npy — array of shape (T', embedding_dim)
  * artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}_t_end.npy — matching t_end for each embedding

These files form the only contract Stage 2 needs: per-session sequences of (z_t, t_end).

This Stage 1 design must support swapping encoders, objectives, window lengths, hops, and preprocessing knobs purely via config, without changing the base data layer or Stage 2 code. 


Stage 2: Clustering, segmentation, and latent-space diagnostics

Goal. Take the encoder embeddings (z_t) (one per window, aligned to t_end) and turn them into:

* a discrete state sequence (cluster IDs) per session, and
* variable-length driving segments (contiguous runs of a state),
  with enough diagnostics to check whether the latent space is actually “cluster-friendly” before trusting any downstream analysis. This stage must support multiple clustering algorithms (k-means, TICC, …) under a common interface.

Inputs and assumptions

* Inputs come only from Stage 1:

  * embeddings: `artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}.npy` — shape (T', embedding_dim)
  * t_end indices: `artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}_t_end.npy` — shape (T',)
* Clustering is fit on **train** embeddings only; val/test are used for evaluation and visualization.

Stage 2.0 – Latent-space diagnostics (pre-clustering)

Before fitting any clustering model, run quick diagnostics on the **train** embeddings to check whether the latent space shows structure:

* Sampling and PCA:

  * Optionally subsample train embeddings (e.g. up to N_diag points).
  * Fit PCA on train embeddings; inspect:

    * explained variance per component,
    * 2D/3D PCA scatter plots (optionally colored by speed or other simple scalar from meta, if available later).
* Clusterability metrics (for k-means-like models):

  * For a range of candidate K (e.g. 4–32), compute:

    * inertia / within-cluster sum of squares,
    * silhouette scores on a subsample.
  * Save these curves for later (e.g. `diagnostics/inertia_vs_k.csv`, `diagnostics/silhouette_vs_k.csv`).
* Basic sanity checks:

  * per-dimension variance of z (detect collapsed embeddings),
  * distribution of (|z_t|) (norms) to spot outliers.

These diagnostics don’t change any downstream behavior but should be run at least once per encoder run to decide if clustering is meaningful at all.

Stage 2.1 – Clustering model interface

Clustering algorithms live in a common module namespace, e.g.:

* `traceability/clustering/`

  * `kmeans.py`
  * `ticc.py`
  * (future) `gmm.py`, `hmm.py`, …

Each clustering implementation is registered via a small registry and must implement a common interface:

* `fit(Z_train) -> model`

  * Z_train: 2D array of shape (N_train_windows, embedding_dim), pooled over all train sessions.
* `predict(Z) -> cluster_id_raw`

  * Z: 2D array of embeddings from any split.
  * Returns `cluster_id_raw`: 1D array of integer cluster IDs.

Configuration:

* The Stage 2 config specifies:

  * `clusterer.name` — e.g. `"kmeans"`, `"ticc"`.
  * `clusterer.kwargs` — passed through to the implementation:

    * For KMeans: `n_clusters`, `random_state`, `n_init`, `max_iter`, `pca_dim` (optional PCA pre-projection), etc.
    * For TICC: window size, regularization, temporal smoothness, etc.
* The driver script (e.g. `run_stage2.py`) is agnostic to which clustering algorithm is used; it just calls into the registry using `clusterer.name`.

K-means baseline (paper-aligned):

* For k-means, Stage 2.1:

  * pools all **train** embeddings from all sessions into one array Z_train,
  * (optionally) applies PCA to dimension `clusterer.kwargs.pca_dim` before fitting,
  * fits a single global KMeans with `n_clusters = K`.

Stage 2.2 – Per-window cluster assignment and smoothing

For each split in `--splits train,val,test` and each session:

1. Load embeddings and t_end:

   * Z_session: `artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}.npy` — shape (T', embedding_dim)
   * t_end: `artifacts/stage1/{encoder_name}/{objective_name}/{run_id}/embeddings/{split}/{session_id}_t_end.npy` — shape (T',)
   * Define `window_idx` as 0..T'-1 for that session.

2. Raw cluster IDs (no temporal smoothing):

   * `cluster_id_raw = model.predict(Z_session)` — 1D array of length T'.

3. Temporal smoothing (optional; implemented as a pluggable post-processor)

   * Smoothing is an optional post-processing step applied to `cluster_id_raw`
     to reduce flicker; it must NOT change the length or alignment (still one ID per window).
   * Configure via:

     * `smooth.name` (e.g., "none", "majority")
     * `smooth.kwargs` (algorithm-specific params)

   * Default smoother: majority / mode filter

     * window length `smooth.window_size` must be odd (e.g., 5)
     * boundary handling: pad by edge replication
     * replace each position with the mode of its neighborhood
     * tie-break rule: if multiple modes, keep the center value

   * Optional minimum segment length enforcement

     * compute run-length encoding on the smoothed sequence
     * any segment shorter than `smooth.min_segment_len` windows is merged into
       an adjacent segment (default rule: merge into the neighbor with the longer duration;
       if both equal, merge into the previous segment)

   * The result is `cluster_id_smooth` — same shape and alignment as `cluster_id_raw`.

4. Save **per-window assignments** to a single CSV per split:

   * `artifacts/stage2/<run_tag>/assignments_{split}.csv`
   * At minimum, columns:

     * `session_id`
     * `window_idx`       # 0-based index in the embedding sequence for that session
     * `t_end`            # index into the original 3 Hz time axis
     * `cluster_id_raw`
     * `cluster_id_smooth`
   * Optional but useful extras:

     * `t_end_s` (seconds) = `t_end / fs`
     * any simple meta signals if available later (e.g. mean speed in the window).

Stage 2.3 – Segmentation via cluster changepoints

segmentation is purely driven by changepoints in the (smoothed) cluster sequence:

* For each session and split:

  * Take `cluster_id_smooth[0..T'-1]`.
  * Run-length encode this sequence:

    * each maximal run of a constant `cluster_id_smooth` becomes one segment.

* For each segment:

  * `segment_no`: 0-based index per session (order of appearance).
  * `segment_start`: window_idx of the first window in the segment.
  * `segment_end`: window_idx of the last window in the segment.
  * `cluster_id_smooth`: constant cluster ID for that segment.

  * Optionally derive:

    * `t_start_end = t_end[segment_start]`          # end-index of first window in segment
    * `t_end_end   = t_end[segment_end]`            # end-index of last window in segment
    * `duration_windows = segment_end - segment_start + 1`
    * `duration_sec = (t_end[segment_end] - t_end[segment_start]) / fs`.

* Save **segment-level summaries** to a CSV per split:

  * `artifacts/stage2/<run_tag>/segments_{split}.csv`
  * At minimum, columns:

    * `session_id`
    * `segment_no`
    * `segment_start`
    * `segment_end`
    * `cluster_id_smooth`
  * Optional extras (recommended):

    * `t_start`, `t_end`, `duration_sec`, `duration_windows`.

This is the “simultaneous clustering and segmentation” described in the paper: the same cluster assignment sequence both defines segments and clusters them (each segment’s label is its constant cluster id).

Stage 2.4 – Driving event discovery and diagnostics

Once assignments and segments exist, Stage 2 (or a lightweight analysis script) should provide:

- Cluster-level summaries:
  * cluster sizes (number of windows and number of segments per cluster),
  * segment duration distributions per cluster,
  * example segments per cluster for qualitative inspection.

- Timeline visualizations:
  * for a given session, a simple plot of `cluster_id_smooth` over time,
  * optionally vertical lines marking segment boundaries,

These diagnostics are not strictly required for the Stage 2 contract but are strongly recommended to assess whether clusters correspond to meaningful driving regimes (braking, accelerating, cruising, turning, etc.). 




### Additional design contracts
* Path conventions are defined in docs/PATHS.md and are mandatory.
* Do not hardcode paths in code. Always use traceability/utils/paths.py.

### Instrumentation
* Every Stage 1 experiment run MUST write TensorBoard logs under <run_dir>/tensorboard/.
* TensorBoard logging requirements and tag conventions are defined in docs/TENSORBOARD.md and are mandatory.

### Config schema (mandatory)
* The canonical experiment config schema and examples are defined in docs/CONFIGS.md.
* All objectives must read their hyperparameters only from objective.kwargs (no predictive.* / contrastive.* / masked.* namespaces).
* All runs must write a resolved config.json (same content as the loaded config with defaults applied).

