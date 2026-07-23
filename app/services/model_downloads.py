"""
Local model weight downloads and detection.

Downloads run as durable `model.download` jobs on the existing job queue and
land in the packed weights folder (`app/model_weights`, or `LOCAL_WEIGHTS_DIR`
when set) using the same directory naming the runners resolve at load time.
Detection scans that folder so the UI can show which catalog models are
already present and surface any extra weight directories it finds.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from agents.model_catalog import MODEL_SPECS, ModelSpec, get_model_spec
from app.services.job_queue import job_handler
from app.services.provider_keys import resolve_huggingface_token


logger = logging.getLogger(__name__)

DOWNLOAD_JOB_KIND = "model.download"

# Catalog IDs whose Hugging Face repository differs from the ID itself.
HF_REPO_OVERRIDES: dict[str, str] = {
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# Directories that predate per-model naming: both Llama 3.1 IDs load from the
# packed llama_3_1 folder (see mlx_llama_runner and transformers_runner).
WEIGHTS_DIR_OVERRIDES: dict[str, str] = {
    "Meta-Llama-3.1-8B-Instruct": "llama_3_1",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama_3_1",
}

_WEIGHT_MARKERS = ("config.json", "params.json", "tokenizer.json")


def weights_root() -> str:
    """The packed weights folder runners load from."""
    local_root = os.environ.get("LOCAL_WEIGHTS_DIR")
    if local_root:
        return os.path.expanduser(local_root)
    return os.path.join("app", "model_weights")


def model_dir_name(model_id: str) -> str:
    """Directory component for a model ID, matching runner conventions."""
    override = WEIGHTS_DIR_OVERRIDES.get(model_id)
    if override:
        return override
    directory_name = re.sub(r"[\\/]+", "_", model_id.strip()).strip(".")
    if not directory_name:
        raise ValueError("Model ID must contain a directory-safe name")
    return directory_name


def model_weights_dir(model_id: str) -> str:
    return os.path.join(weights_root(), model_dir_name(model_id))


def hf_repo_id(model_id: str) -> str:
    """Hugging Face repository to download for a catalog model ID."""
    return HF_REPO_OVERRIDES.get(model_id, model_id)


def is_model_downloaded(model_id: str) -> bool:
    return _dir_has_weights(model_weights_dir(model_id))


def _dir_has_weights(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    if any(marker in entries for marker in _WEIGHT_MARKERS):
        return True
    return any(entry.endswith((".safetensors", ".gguf", ".pth")) for entry in entries)


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def downloadable_model_specs() -> list[ModelSpec]:
    """Catalog models that can be downloaded into the packed weights folder."""
    return [spec for spec in MODEL_SPECS if spec.provider == "offline" and spec.local]


def _latest_download_jobs(user_id: int | None) -> dict[str, dict[str, Any]]:
    """Newest download job per model ID, keyed by dedupe model ID."""
    from app.models.database.database import SessionLocal
    from app.models.database.job import Job

    jobs: dict[str, dict[str, Any]] = {}
    try:
        with SessionLocal() as session:
            query = session.query(Job).filter(Job.kind == DOWNLOAD_JOB_KIND)
            if user_id is not None:
                query = query.filter(Job.user_id == user_id)
            for job in query.order_by(Job.job_id.asc()).all():
                job_dict = job.to_dict()
                model_id = (job_dict.get("payload") or {}).get("model_id")
                if model_id:
                    jobs[model_id] = job_dict
    except Exception:
        logger.warning("Could not read model download jobs", exc_info=True)
    return jobs


def local_model_statuses(user_id: int | None = None) -> dict[str, Any]:
    """
    Describe downloadable catalog models plus any extra weight directories.

    Returns {"models": [...], "detected_directories": [...], "weights_root": str}.
    """
    root = weights_root()
    jobs = _latest_download_jobs(user_id)

    models = []
    known_dirs = set()
    for spec in downloadable_model_specs():
        directory = model_weights_dir(spec.id)
        known_dirs.add(os.path.basename(directory))
        downloaded = _dir_has_weights(directory)
        job = jobs.get(spec.id)
        models.append({
            "id": spec.id,
            "name": spec.name,
            "family": spec.family,
            "backend": spec.backend,
            "gated": spec.gated,
            "parameter_count": spec.parameter_count,
            "downloaded": downloaded,
            "weights_path": directory,
            "size_bytes": _dir_size_bytes(directory) if downloaded else 0,
            "download_status": job.get("status") if job else None,
            "download_job_id": job.get("job_id") if job else None,
            "download_error": job.get("error") if job else None,
        })

    detected = []
    if os.path.isdir(root):
        for entry in sorted(os.listdir(root)):
            path = os.path.join(root, entry)
            if entry in known_dirs or not _dir_has_weights(path):
                continue
            detected.append({
                "directory": entry,
                "weights_path": path,
                "size_bytes": _dir_size_bytes(path),
            })

    return {"models": models, "detected_directories": detected, "weights_root": root}


def enqueue_model_download(
    model_id: str,
    user_id: int,
    revision: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Validate and queue one model download; returns the queued job as a dict.

    Only catalog models marked offline/local can be queued, so heavyweight
    server-backed IDs can never be pulled onto the local disk by the UI.
    """
    from app.models.database.database import SessionLocal
    from app.models.database.job import Job, JobStatus
    from app.services.job_queue import enqueue

    spec = get_model_spec(model_id)
    if spec is None or spec.provider != "offline" or not spec.local:
        raise ValueError(f"'{model_id}' is not a downloadable local catalog model")
    if is_model_downloaded(spec.id) and not force:
        raise ValueError(f"'{spec.id}' is already downloaded")
    if spec.gated and not resolve_huggingface_token(user_id):
        raise ValueError(
            f"'{spec.id}' is a gated model: add a Hugging Face token under "
            "Settings > Providers first"
        )

    dedupe_key = f"{DOWNLOAD_JOB_KIND}:{spec.id}"
    with SessionLocal() as session:
        active = (
            session.query(Job)
            .filter(
                Job.dedupe_key == dedupe_key,
                Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
            )
            .first()
        )
        if active is not None:
            return active.to_dict()

    job = enqueue(
        DOWNLOAD_JOB_KIND,
        payload={"model_id": spec.id, "revision": revision, "user_id": user_id},
        max_attempts=2,
        user_id=user_id,
        dedupe_key=dedupe_key,
    )
    return job.to_dict()


@job_handler(DOWNLOAD_JOB_KIND)
def _run_model_download_job(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Download one model's weights from Hugging Face into the packed folder.

    Payload: {"model_id": str, "revision": str | None, "user_id": int | None},
    enqueued exclusively by enqueue_model_download after catalog validation.
    """
    # Imported here so the API process never pays the huggingface_hub import.
    from huggingface_hub import snapshot_download

    model_id = payload["model_id"]
    revision = payload.get("revision")
    user_id = payload.get("user_id")

    spec = get_model_spec(model_id)
    if spec is None or spec.provider != "offline" or not spec.local:
        raise ValueError(f"'{model_id}' is not a downloadable local catalog model")

    destination = model_weights_dir(spec.id)
    token = resolve_huggingface_token(user_id)
    logger.info("Downloading %s to %s", hf_repo_id(spec.id), destination)
    snapshot_download(
        repo_id=hf_repo_id(spec.id),
        local_dir=destination,
        token=token,
        revision=revision,
    )
    return {
        "model_id": spec.id,
        "weights_path": destination,
        "size_bytes": _dir_size_bytes(destination),
    }
