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
- Integer scenario IDs. Globally there are 12 labels in the dataset, but each session may contain only a subset.
- Labels are aligned to the same 3 Hz timeline as sensors.
- Labels must be used only for evaluation (cluster-label agreement, qualitative inspection).

Core method to implement (paper-aligned)
We treat each window as a subsequence ending at an index t_end:
- window length w (timesteps)
- X_window = X_session[t_end-(w-1) : t_end+1]  shape (w, 8)
We attach the embedding z_t to the end time t_end (timestamp-level representation).
Then per session we create an embedding sequence Z_session in time order, cluster embeddings to obtain a discrete state sequence, and segment by changepoints where cluster id changes.

Windowing configuration (initial agreed settings)
Sampling rate fs = 3 Hz
Window length L = 6 seconds -> w = 18 timesteps
Training stride = 1 second -> hop_train = 3 timesteps
Inference/clustering stride recommendation -> hop_infer = 1 timestep (dense, 0.33 s resolution)
Rationale: training uses fewer overlapping windows to reduce redundancy; inference uses dense windows for better segmentation.

Unified dataset design (do not duplicate data)
Do not pre-slice and save all windows to disk.
Instead, build an index that points into the existing session arrays.

Index entry definition
Minimal index entry:
- session_id, t_end
Derived start index:
- t_start = t_end - (w - 1)

Two separate indices are typically needed
- train index: uses hop_train = 3
- infer index: uses hop_infer = 1
Both indices should be reproducible with fixed parameters and seed.

Splitting strategy (critical)
Split by session_id, never by windows, to avoid leakage due to overlapping windows from the same drive.
Recommended: 70/15/15 train/val/test split by session IDs with a fixed random seed.
After splitting sessions, build window indices separately for each split.

Normalization and preprocessing
Keep preprocessing minimal initially.
Compute per-channel mean and std on train split only (across all train windows or across all timesteps in train sessions).
Apply the same normalization to val/test.
Binary channels lturn/rturn can remain 0/1 or be standardized; pick one approach and keep it consistent.
Handle NaNs if present: log which sessions contain NaNs, then decide to drop affected windows or forward-fill.

What the dataset must support (future-proof)
Training stage (encoder learning)
- Return x window of shape (w, 8)
- Optionally return two augmented views x1, x2 if contrastive SSL is used later
- Return metadata session_id and t_end (metadata is not fed to the model; it is for bookkeeping)

Inference stage (embedding extraction for clustering)
- Iterate windows in time order per session using infer index
- Produce embedding sequence Z_session aligned by t_end
- Cluster embeddings to get cluster_id[t_end]
- Segment by changepoints in the cluster id sequence

Label alignment for evaluation
Simplest alignment: assign window label as y[t_end] (label at window end).
Alternative: majority label over the window or center label; keep this configurable.

Deliverables expected from the agent (implementation plan)
1) Dataset scan and validation
- Iterate all files in 20200710_sensors/sensor and 20200710_labels/target
- Verify each session_id exists in both folders
- Verify X.shape[0] == y.shape[0] per session
- Report dataset summary: number of sessions, lengths, duration in minutes, feature dtype, label inventory, per-label counts, per-session label diversity

2) Session split builder
- Build train/val/test lists of session_ids with a fixed seed
- Save splits to a small file, e.g. artifacts/splits.json

3) Window index builder
- Input: session_ids, w, hop
- Output: list of (session_id, t_end)
- Save indices, e.g. artifacts/index_train.jsonl, artifacts/index_val.jsonl, artifacts/index_test.jsonl
- Build both training and inference indices if requested

4) Dataset class
- Uses an index list and loads the needed session file, slices windows on demand
- Supports optional label return for evaluation
- Applies normalization consistently
- Must be efficient: consider np.load with mmap_mode="r" and caching recently used sessions

5) Quick sanity checks
- Randomly sample a few windows, verify shape (18,8)
- Verify time alignment: label slicing matches sensor slicing lengths
- Plot a few channels for a random session to confirm signals look sane

Practical quickstart (what to run first)
- Run the dataset scan to confirm shapes and matching session IDs
- Create session splits
- Build train index using w=18 and hop_train=3
- Build infer index using w=18 and hop_infer=1
- Implement dataset class and iterate a small batch to confirm correct window extraction

Notes and pitfalls
- Do not add timestamps to the sensor arrays for this 3 Hz subset; time is implicit.
- Do not shuffle windows across sessions for splitting; always split by session_id.
- Inference windows must be generated in time order per session for segmentation to make sense.
- Labels are evaluation-only; no supervised learning should be baked into the training objective in this phase.

North star outcomes for this phase
- Build a working pipeline: windows -> encoder embeddings -> clustering -> segments
- Show that clusters correspond to distinct driving regimes (e.g., braking vs accelerating vs turning vs cruising) via cluster prototypes and temporal coherence
- Use labels only to quantify sanity (purity/NMI) and to debug, not to train

Stage 1: Autoencoder baseline (self-supervised)
- Train (sparse windows, hop=3):
  `python train_encoder.py --train_index artifacts/index_train_sparse.jsonl --val_index artifacts/index_val_sparse.jsonl --epochs 50 --batch_size 128`
- Outputs in `artifacts/encoder/`:
  - `weights_last.pt`, `weights_best.pt`
  - `config.json` (hyperparams, window, embedding_dim, seed)
  - `metrics.json` (train/val losses)
- Extract embeddings (dense windows, hop=1):
  `python extract_embeddings.py --splits train,val,test`
- Outputs in `artifacts/embeddings/{split}/`:
  - `{session_id}.npy` (shape: T' x e)
  - `{session_id}_t_end.npy` (shape: T', window end indices)

Stage 2: Clustering + segmentation (paper-aligned)
- Stage 2.1 (KMeans): fit on train embeddings only, predict cluster id per window.
- Stage 2.2 (Segmentation): split each per-session sequence at every cluster-id change (run-length encoding).
- Run Stage 2:
  `python run_stage2.py --embeddings_dir artifacts/embeddings --index_dir artifacts --out_dir artifacts/stage2 --k 10 --seed 123 --splits train,val,test`
- Outputs in `artifacts/stage2/kmeans_k{K}_seed{SEED}/`:
  - `kmeans.joblib` (or `kmeans.pkl`), `cluster_centers.npy`
  - `assignments_{split}.csv` (session_id, window_idx, cluster_id, t_end, t_end_s)
  - `seq_by_session_{split}.json`
  - `segments_{split}.csv`, `segments_by_session_{split}.json`
  - `config.json`, `metrics.json`
  - `viz/{session_id}.png` (cluster seq + changepoints)
