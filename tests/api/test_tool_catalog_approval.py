from types import SimpleNamespace

from fastapi.testclient import TestClient


def test_catalog_exposes_non_waivable_approval(monkeypatch, tmp_path):
    from app import main
    from app.services.tool_registry import build_default_tool_registry

    monkeypatch.setenv("GEIST_EXEC_BACKEND", "local")
    monkeypatch.setenv("GEIST_MARKDOWN_ROOT", str(tmp_path))
    monkeypatch.setenv("GEIST_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.delenv("GEIST_EXEC_WORKSPACE", raising=False)
    monkeypatch.setattr(main, "get_default_workspace", lambda: SimpleNamespace(workspace_id=1))
    monkeypatch.setattr(
        main, "chat_orchestrator", SimpleNamespace(registry=build_default_tool_registry())
    )
    response = TestClient(main.create_app(), client=("127.0.0.1", 50000)).get("/agent/tools")
    assert response.status_code == 200
    tool = next(tool for tool in response.json()["tools"] if tool["name"] == "terminal.run")
    assert tool["requires_per_call_approval"] is True
    assert tool["requires_approval"] is True
