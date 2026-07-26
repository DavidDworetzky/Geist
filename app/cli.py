"""Command-line entry point for the native Geist application."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import socket
import sys
import threading
from collections.abc import Callable, Sequence
from typing import Any, BinaryIO, TextIO

from app.runtime_config import (
    RuntimePaths,
    application_version,
    doctor_report,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5001
DEFAULT_DATABASE_PROVIDER = "sqlite"
MANAGED_PROTOCOL_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geist", description="Run and inspect Geist")
    parser.add_argument("--version", action="version", version=application_version())
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the Geist API and compiled web app")
    serve.add_argument("--host", default=os.getenv("GEIST_HOST", DEFAULT_HOST))
    serve.add_argument("--port", type=_port_number)
    serve.add_argument("--data-dir")
    serve.add_argument(
        "--database-provider",
        choices=("sqlite", "postgresql"),
        default=os.getenv("GEIST_DATABASE_PROVIDER", DEFAULT_DATABASE_PROVIDER),
    )
    web_mode = serve.add_mutually_exclusive_group()
    web_mode.add_argument("--web-dir", help="path to the compiled React build")
    web_mode.add_argument("--api-only", action="store_true", help="do not serve the React app")
    serve.add_argument("--managed-stdio", action="store_true")
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--skip-db-upgrade", action="store_true")
    serve.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.getenv("GEIST_LOG_LEVEL", "info"),
    )
    serve.set_defaults(handler=_serve_command)

    doctor = commands.add_parser("doctor", help="inspect the native runtime")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument("--data-dir")
    doctor.add_argument("--web-dir")
    doctor.set_defaults(handler=_doctor_command)

    upgrade = commands.add_parser("upgrade-db", help="initialize or upgrade the Geist database")
    upgrade.add_argument("--data-dir")
    upgrade.add_argument(
        "--database-provider",
        choices=("sqlite", "postgresql"),
        default=os.getenv("GEIST_DATABASE_PROVIDER", DEFAULT_DATABASE_PROVIDER),
    )
    upgrade.set_defaults(handler=_upgrade_database_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (argparse.ArgumentTypeError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"geist: {error}", file=sys.stderr, flush=True)
        return 2


def _serve_command(args: argparse.Namespace) -> int:
    if not args.allow_remote and not _is_loopback(args.host):
        raise ValueError(
            f"refusing non-loopback host {args.host!r}; pass --allow-remote to expose Geist"
        )

    port = args.port
    if port is None:
        configured_port = os.getenv("GEIST_PORT")
        if configured_port:
            port = _port_number(configured_port)
        elif args.managed_stdio:
            port = 0
        else:
            port = DEFAULT_PORT

    paths = RuntimePaths.resolve(
        data_dir=args.data_dir,
        web_dir=None if args.api_only else args.web_dir,
        require_web=bool(args.web_dir),
    )
    if args.api_only:
        paths = RuntimePaths(
            data_dir=paths.data_dir,
            database_path=paths.database_path,
            output_dir=paths.output_dir,
            model_cache_dir=paths.model_cache_dir,
            web_dir=None,
        )
    paths.apply()
    os.environ["GEIST_DATABASE_PROVIDER"] = args.database_provider

    if not args.skip_db_upgrade:
        _upgrade_database()

    app = _create_application(paths.web_dir, loopback_only=not args.allow_remote)
    return _run_server(
        app,
        host=args.host,
        port=port,
        log_level=args.log_level,
        managed_stdio=args.managed_stdio,
        spa_enabled=paths.web_dir is not None,
    )


def _doctor_command(args: argparse.Namespace) -> int:
    paths = RuntimePaths.resolve(data_dir=args.data_dir, web_dir=args.web_dir)
    report = doctor_report(paths)
    if args.as_json:
        print(json.dumps(report, separators=(",", ":")), flush=True)
    else:
        status = "ok" if report["ok"] else "not ready"
        path_report = report["paths"]
        assert isinstance(path_report, dict)
        print(f"Geist {report['version']}: {status}")
        print(f"Python: {report['python']}")
        print(f"Data: {path_report['dataDir']}")
        print(f"Database: {path_report['databasePath']}")
        print(f"Web: {path_report['webDir'] or 'not built'}")
    return 0 if report["ok"] else 1


def _upgrade_database_command(args: argparse.Namespace) -> int:
    paths = RuntimePaths.resolve(data_dir=args.data_dir)
    paths.apply()
    os.environ["GEIST_DATABASE_PROVIDER"] = args.database_provider
    _upgrade_database()
    print(f"Geist database is ready at {paths.database_path}", flush=True)
    return 0


def _upgrade_database() -> None:
    from app.database_upgrade import upgrade_database

    upgrade_database()


def _create_application(web_dir, *, loopback_only: bool = False):
    from app.main import create_app

    return create_app(web_dir=web_dir, loopback_only=loopback_only)


def _run_server(
    app: Any,
    *,
    host: str,
    port: int,
    log_level: str,
    managed_stdio: bool,
    spa_enabled: bool,
) -> int:
    import uvicorn

    bound_socket = _bind_socket(host, port)
    bound_host, bound_port = bound_socket.getsockname()[:2]
    origin_host = _origin_host(str(bound_host))
    config = uvicorn.Config(
        app,
        host=host,
        port=int(bound_port),
        log_level=log_level,
        reload=False,
    )
    server = uvicorn.Server(config)

    if managed_stdio:
        _start_stdin_eof_watcher(server)

    def announce_ready() -> None:
        event = _readiness_event(
            origin=f"http://{origin_host}:{bound_port}",
            spa_enabled=spa_enabled,
            managed_stdio=managed_stdio,
        )
        print(json.dumps(event, separators=(",", ":")), flush=True)

    try:
        started = asyncio.run(_serve_until_exit(server, bound_socket, announce_ready))
    finally:
        bound_socket.close()
    return 0 if started else 1


def _readiness_event(
    *,
    origin: str,
    spa_enabled: bool,
    managed_stdio: bool,
) -> dict[str, object]:
    return {
        "event": "geist.ready",
        "protocol": MANAGED_PROTOCOL_VERSION,
        "version": application_version(),
        "origin": origin,
        "pid": os.getpid(),
        "capabilities": {
            "spa": spa_enabled,
            "stdinShutdown": managed_stdio,
        },
    }


async def _serve_until_exit(
    server: Any,
    bound_socket: socket.socket,
    announce_ready: Callable[[], None],
) -> bool:
    serve_task = asyncio.create_task(server.serve(sockets=[bound_socket]))
    while not server.started and not serve_task.done():
        await asyncio.sleep(0.01)

    started = bool(server.started)
    if started:
        announce_ready()
    await serve_task
    return started


def _start_stdin_eof_watcher(
    server: Any,
    stream: BinaryIO | TextIO | None = None,
) -> threading.Thread:
    input_stream = stream if stream is not None else sys.stdin.buffer

    def watch() -> None:
        try:
            while input_stream.read(4096):
                pass
        except (OSError, ValueError):
            pass
        finally:
            server.should_exit = True

    thread = threading.Thread(target=watch, name="geist-stdin-eof", daemon=True)
    thread.start()
    return thread


def _bind_socket(host: str, port: int) -> socket.socket:
    address_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not address_info:
        raise OSError(f"could not resolve host {host!r}")
    family, socket_type, protocol, _, socket_address = address_info[0]
    bound_socket = socket.socket(family, socket_type, protocol)
    try:
        bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bound_socket.bind(socket_address)
        bound_socket.set_inheritable(True)
    except Exception:
        bound_socket.close()
        raise
    return bound_socket


def _origin_host(host: str) -> str:
    if ":" in host:
        return f"[{host}]"
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _port_number(value: str | int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


if __name__ == "__main__":
    raise SystemExit(main())
