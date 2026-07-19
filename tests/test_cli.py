import asyncio
import io
import json
import os
import socket

from app import cli


def test_managed_serve_defaults_to_loopback_and_ephemeral_port(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.delenv("GEIST_PORT", raising=False)
    monkeypatch.delenv("GEIST_DATABASE_PROVIDER", raising=False)
    monkeypatch.setattr(cli, "_upgrade_database", lambda: None)
    monkeypatch.setattr(
        cli,
        "_create_application",
        lambda web_dir, *, loopback_only=False: ("app", web_dir, loopback_only),
    )

    def fake_run_server(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "_run_server", fake_run_server)

    result = cli.main(
        ["serve", "--managed-stdio", "--api-only", "--data-dir", str(tmp_path)]
    )

    assert result == 0
    assert captured == {
        "app": ("app", None, True),
        "host": "127.0.0.1",
        "port": 0,
        "log_level": "info",
        "managed_stdio": True,
        "spa_enabled": False,
    }
    assert os.environ["GEIST_DATABASE_PROVIDER"] == "sqlite"


def test_remote_bind_requires_an_explicit_opt_in(tmp_path, capsys):
    result = cli.main(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--api-only",
            "--skip-db-upgrade",
            "--data-dir",
            str(tmp_path),
        ]
    )

    assert result == 2
    assert "refusing non-loopback host" in capsys.readouterr().err


def test_readiness_event_matches_managed_protocol(monkeypatch):
    monkeypatch.setattr(cli, "application_version", lambda: "9.8.7")
    monkeypatch.setattr(os, "getpid", lambda: 123)

    event = cli._readiness_event(
        origin="http://127.0.0.1:43210",
        spa_enabled=True,
        managed_stdio=True,
    )

    assert event == {
        "event": "geist.ready",
        "protocol": 1,
        "version": "9.8.7",
        "origin": "http://127.0.0.1:43210",
        "pid": 123,
        "capabilities": {"spa": True, "stdinShutdown": True},
    }
    assert json.loads(json.dumps(event)) == event


def test_stdin_eof_requests_server_shutdown():
    server = type("Server", (), {"should_exit": False})()

    watcher = cli._start_stdin_eof_watcher(server, stream=io.BytesIO(b""))
    watcher.join(timeout=1)

    assert not watcher.is_alive()
    assert server.should_exit is True


def test_readiness_is_announced_after_server_startup():
    events = []

    class FakeServer:
        started = False

        async def serve(self, sockets):
            assert len(sockets) == 1
            await asyncio.sleep(0)
            self.started = True
            await asyncio.sleep(0.01)

    with socket.socket() as bound_socket:
        started = asyncio.run(
            cli._serve_until_exit(FakeServer(), bound_socket, lambda: events.append("ready"))
        )

    assert started is True
    assert events == ["ready"]
