"""Submit Azure ML jobs from YAML with optional overrides."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote

from azure.ai.ml import load_job

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ws import get_ml_client


def _studio_url(settings: dict[str, str], experiment_name: str, job_name: str) -> str:
    sub = quote(settings["subscription_id"], safe="")
    rg = quote(settings["resource_group"], safe="")
    ws = quote(settings["workspace_name"], safe="")
    exp = quote(experiment_name, safe="")
    job = quote(job_name, safe="")
    wsid = f"/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.MachineLearningServices/workspaces/{ws}"
    return f"https://ml.azure.com/experiments/{exp}/runs/{job}?wsid={wsid}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an Azure ML job from YAML.")
    parser.add_argument("--job", required=True, help="Path to a job YAML file.")
    parser.add_argument("--compute", default=None, help="Override compute target.")
    parser.add_argument("--environment", default=None, help="Override environment (azureml:NAME@VER).")
    parser.add_argument("--experiment_name", default=None, help="Override experiment name.")
    args = parser.parse_args()

    client, settings = get_ml_client()
    job = load_job(args.job)

    if args.compute:
        job.compute = args.compute
    if args.environment:
        job.environment = args.environment
    if args.experiment_name:
        job.experiment_name = args.experiment_name

    created = client.jobs.create_or_update(job)
    print(f"Job name: {created.name}")

    studio_url = getattr(created, "studio_url", None)
    if not studio_url:
        exp_name = created.experiment_name or args.experiment_name or "Default"
        studio_url = _studio_url(settings, exp_name, created.name)
    print(f"Studio URL: {studio_url}")


if __name__ == "__main__":
    main()
