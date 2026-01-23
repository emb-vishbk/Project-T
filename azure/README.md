# Azure ML workflow

This folder contains Azure-specific helpers only. Core training/data/model code stays Azure-agnostic.

## 1) Setup (local)
Activate your Azure venv and install SDKs:

```powershell
.\.venv_azure\Scripts\Activate.ps1
python -m pip install azure-ai-ml azure-identity
```

## 2) Workspace config
Edit `azure/config.json` with your workspace info. For secrets or personal values,
use `azure/config.local.json` (ignored by git). The local file overrides `config.json`.

Example:
```json
{
  "subscription_id": "<SUBSCRIPTION_ID>",
  "resource_group": "<RESOURCE_GROUP>",
  "workspace_name": "<WORKSPACE_NAME>"
}
```

## 3) List curated environments
```powershell
python azure/scripts/list_envs.py --contains pytorch
```
Pick a curated environment string like `azureml:AzureML-pytorch-...@latest`.

## 4) Submit jobs
Smoke test (no data asset required):
```powershell
python azure/scripts/submit_job.py --job azure/jobs/smoke_test.yaml --compute <COMPUTE_NAME> --environment azureml:<ENV_NAME>@latest
```

Data-mount smoke test (requires a Data Asset that contains the HDD data):
```powershell
python azure/scripts/submit_job.py --job azure/jobs/data_mount_smoke.yaml --compute <COMPUTE_NAME> --environment azureml:<ENV_NAME>@latest
```

Stage 1 train (data prep + training + embeddings):
```powershell
python azure/scripts/submit_job.py --job azure/jobs/stage1_train.yaml --compute <COMPUTE_NAME> --environment azureml:<ENV_NAME>@latest
```

## 5) Stream logs / status
```powershell
python azure/scripts/stream_job.py --job_name <JOB_NAME>
python azure/scripts/stream_job.py --job_name <JOB_NAME> --status_only
```

## 6) Download outputs
```powershell
python azure/scripts/download_job.py --job_name <JOB_NAME> --output_name artifacts_root
```

## Notes
- `azure/jobs/*.yaml` use curated environments only (no conda files).
- Update `azure/jobs/*.yaml` placeholders:
  - `compute: gpu-cluster`
  - `environment: azureml:...@latest`
  - `inputs.data_root.path: azureml:REPLACE_ME_DATA_ASSET@latest`
- `data_mount_smoke.yaml` and `stage1_train.yaml` expect `inputs.data_root` to point
  to a folder that contains `raw/` with:
  - `raw/20200710_sensors/sensor/*.npy`
  - `raw/20200710_labels/target/*.npy`
- All generated outputs are written under `outputs.artifacts_root` (including
  `data_info/meta` and `data_info/index`).
