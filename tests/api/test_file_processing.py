import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.utils import get_current_workspace
from app.api.v1.endpoints import files, jobs
from app.models.database.file_upload import get_file_upload_by_id
from app.models.database.geist_user import ensure_default_workspace
from app.models.database.job import get_job
from app.services.file_processor import FileProcessingService
from app.services.job_queue import JobWorker, _run_file_processing_job
from tests.services.test_job_worker import sqlite_database as sqlite_database


@pytest.fixture
def file_app(request):
    request.getfixturevalue("sqlite_database")
    workspace = ensure_default_workspace()
    app = FastAPI()
    app.include_router(files.router, prefix="/files")
    app.include_router(jobs.router, prefix="/jobs")
    app.dependency_overrides[get_current_workspace] = lambda: workspace
    return app


def test_upload_returns_before_extraction_and_worker_persists_text(file_app, monkeypatch):
    extract = Mock(return_value={"success": True, "text": "processed text"})
    monkeypatch.setattr(
        "app.services.file_processor.TextExtractionService.extract_text_from_file", extract
    )
    with TestClient(file_app) as client:
        response = client.post(
            "/files/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processing_status"]["status"] == "queued"
        extract.assert_not_called()
        assert not get_file_upload_by_id(data["file_id"]).is_processed
        job_id = data["processing_status"]["job_id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "queued"
        assert JobWorker().run_once()
        assert get_file_upload_by_id(data["file_id"]).extracted_text == "processed text"
        result = client.get(f"/jobs/{job_id}").json()
        assert result["status"] == "succeeded"
        assert result["result"] == {"success": True, "file_id": data["file_id"]}
        assert extract.call_count == 1


def test_failed_extraction_is_failed_job_and_reprocess_can_recover(file_app, monkeypatch):
    extract = Mock(return_value={"success": False, "error": "bad document"})
    monkeypatch.setattr(
        "app.services.file_processor.TextExtractionService.extract_text_from_file", extract
    )
    with TestClient(file_app) as client:
        data = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        duplicate = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        JobWorker().run_once()
        JobWorker().run_once()
        assert get_job(duplicate["processing_status"]["job_id"]).status == "failed"
        assert get_job(data["processing_status"]["job_id"]).status == "failed"
        assert (
            client.get(f"/files/{data['file_id']}/status").json()["processing_error"]
            == "bad document"
        )
        extract.return_value = {"success": True, "text": "recovered"}
        retry = client.post(f"/files/{data['file_id']}/reprocess").json()
        assert extract.call_count == 1
        JobWorker().run_once()
        assert get_job(retry["processing_result"]["job_id"]).status == "succeeded"
        assert get_file_upload_by_id(data["file_id"]).extracted_text == "recovered"


def test_worker_rechecks_ownership_and_missing_file(monkeypatch):
    process = Mock()
    monkeypatch.setattr(FileProcessingService, "process_file_sync", process)
    for uploaded in (None, SimpleNamespace(user_id=2)):
        monkeypatch.setattr(
            "app.models.database.file_upload.get_file_upload_by_id",
            lambda _, uploaded=uploaded: uploaded,
        )
        with pytest.raises(ValueError, match="workspace"):
            _run_file_processing_job({"file_id": 1, "user_id": 1})
    process.assert_not_called()


@pytest.mark.asyncio
async def test_upload_storage_does_not_block_event_loop(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def store(**kwargs):
        entered.set()
        assert release.wait(5)
        return {"success": True, "file_id": 1, "filename": "notes.txt", "duplicate": False}

    monkeypatch.setattr(files.FileStorageService, "store_file", store)
    monkeypatch.setattr(
        files, "enqueue_file_processing", lambda *a, **kw: SimpleNamespace(job_id=1)
    )
    app = FastAPI()
    app.include_router(files.router)
    app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(workspace_id=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        request = asyncio.create_task(client.post("/upload", files={"file": ("notes.txt", b"hi")}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # The event loop can resume while the storage thread remains blocked.
            assert not request.done()
        finally:
            release.set()
        assert (await request).status_code == 200


def test_duplicate_waits_for_processing(file_app, monkeypatch):
    with TestClient(file_app) as client:
        data = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        duplicate = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        assert duplicate["duplicate"]
        assert duplicate["processing_status"]["job_id"] == data["processing_status"]["job_id"]
        assert duplicate["file_id"] == data["file_id"]
        worker = JobWorker()
        worker.run_once()
        worker.run_once()
        assert get_job(duplicate["processing_status"]["job_id"]).status == "succeeded"
        completed = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        assert completed["processing_status"] == {}


def test_foreign_workspace_cannot_reprocess_or_observe_job(file_app):
    with TestClient(file_app) as client:
        data = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        file_app.dependency_overrides[get_current_workspace] = lambda: SimpleNamespace(
            workspace_id=999
        )
        assert client.post(f"/files/{data['file_id']}/reprocess").status_code == 403
        assert client.get(f"/jobs/{data['processing_status']['job_id']}").status_code == 404
        from app.models.database.database import SessionLocal
        from app.models.database.geist_user import GeistUser

        with SessionLocal() as session:
            session.add(GeistUser(user_id=999, workspace_key="other"))
            session.commit()
        other = client.post("/files/upload", files={"file": ("notes.txt", b"hello")})
        assert other.status_code == 200
        assert other.json()["file_id"] != data["file_id"]
        assert not other.json()["duplicate"]
        assert get_file_upload_by_id(other.json()["file_id"]).user_id == 999


def test_repeated_upload_reuses_running_job(file_app):
    from app.models.database.job import claim_next_job

    with TestClient(file_app) as client:
        data = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        running = claim_next_job()
        assert running.job_id == data["processing_status"]["job_id"]
        duplicate = client.post("/files/upload", files={"file": ("notes.txt", b"hello")}).json()
        assert duplicate["processing_status"]["job_id"] == running.job_id
        assert claim_next_job() is None
