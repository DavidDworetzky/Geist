from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.static_web import install_spa


def _compiled_web_app(tmp_path):
    web_dir = tmp_path / "build"
    static_dir = web_dir / "static" / "js"
    static_dir.mkdir(parents=True)
    (web_dir / "index.html").write_text("<html>GEIST_SPA</html>", encoding="utf-8")
    (web_dir / "manifest.json").write_text('{"name":"Geist"}', encoding="utf-8")
    (static_dir / "main.123.js").write_text("window.geist = true;", encoding="utf-8")
    return web_dir


def test_compiled_spa_and_browser_routes_share_the_api_origin(tmp_path):
    app = FastAPI()

    @app.get("/api/known")
    def known_api():
        return {"known": True}

    install_spa(app, _compiled_web_app(tmp_path))

    with TestClient(app) as client:
        root = client.get("/")
        browser_route = client.get("/chat/42")
        api = client.get("/api/known")
        manifest = client.get("/manifest.json")
        script = client.get("/static/js/main.123.js")

    assert root.status_code == 200
    assert "GEIST_SPA" in root.text
    assert root.headers["cache-control"] == "no-cache"
    assert browser_route.status_code == 200
    assert "GEIST_SPA" in browser_route.text
    assert api.json() == {"known": True}
    assert manifest.json() == {"name": "Geist"}
    assert "window.geist" in script.text


def test_backend_namespaces_never_fall_back_to_the_spa(tmp_path):
    app = FastAPI()
    install_spa(app, _compiled_web_app(tmp_path))

    with TestClient(app) as client:
        for path in (
            "/api/missing",
            "/agent/missing",
            "/adapter/missing",
            "/health/missing",
            "/openapi.json/missing",
            "/static/missing.js",
        ):
            response = client.get(path)
            assert response.status_code == 404
            assert "GEIST_SPA" not in response.text


def test_branding_defaults_to_an_empty_same_origin_document(tmp_path):
    app = FastAPI()
    install_spa(app, _compiled_web_app(tmp_path))

    with TestClient(app) as client:
        response = client.get("/branding.json")

    assert response.status_code == 200
    assert response.json() == {}
