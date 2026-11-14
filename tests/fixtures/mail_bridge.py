"""Local TLS mail fixture. Uses fake credentials and never contacts external mail."""

import contextlib
import socketserver
import ssl
import subprocess
import threading


@contextlib.contextmanager
def bridge_fixture(tmp_path, *, starttls=False):
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    state = {"commands": [], "messages": [], "disconnect_send": False}

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    class Handler(socketserver.StreamRequestHandler):
        def setup(self):
            if not starttls:
                self.request = context.wrap_socket(self.request, server_side=True)
            super().setup()

        def reply(self, line):
            self.wfile.write(line + b"\r\n")
            self.wfile.flush()

        def upgrade(self):
            self.connection = context.wrap_socket(self.connection, server_side=True)
            self.rfile = self.connection.makefile("rb")
            self.wfile = self.connection.makefile("wb")

        def handle(self):
            smtp = self.server.smtp
            self.reply(b"220 test SMTP" if smtp else b"* OK test IMAP")
            while line := self.rfile.readline():
                # Record only command names and test bodies, never AUTH/LOGIN payloads.
                parts = line.strip().split()
                if not parts:
                    continue
                command = parts[0 if smtp else 1].upper()
                state["commands"].append(command)
                if command == b"STARTTLS":
                    self.reply(b"220 ready" if smtp else parts[0] + b" OK ready")
                    self.upgrade()
                elif smtp:
                    if command == b"EHLO":
                        self.reply(b"250-test\r\n250-STARTTLS\r\n250 AUTH PLAIN")
                    elif command in (b"AUTH",):
                        self.reply(b"235 authenticated")
                    elif command == b"DATA":
                        self.reply(b"354 send data")
                        body = b""
                        while (chunk := self.rfile.readline()) != b".\r\n":
                            if not chunk:
                                return
                            body += chunk
                        state["messages"].append(body)
                        if state["disconnect_send"]:
                            return
                        self.reply(b"250 accepted")
                    elif command == b"QUIT":
                        self.reply(b"221 goodbye")
                        return
                    else:
                        self.reply(b"250 OK")
                else:
                    tag = parts[0]
                    if command == b"CAPABILITY":
                        self.reply(b"* CAPABILITY IMAP4rev1 STARTTLS")
                    elif command == b"LIST":
                        self.reply(b'* LIST (\\HasNoChildren) "/" "INBOX"')
                    elif command in (b"SELECT", b"EXAMINE"):
                        self.reply(b"* 1 EXISTS")
                        self.reply(b"* OK [UIDVALIDITY 123] stable")
                    elif command == b"UID":
                        if parts[2].upper() == b"SEARCH":
                            self.reply(b"* SEARCH 42")
                        elif parts[2].upper() == b"FETCH":
                            data = b"From: alice@example.test\r\nTo: test@example.test\r\nSubject: Fixture message\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nHello from local fixture."
                            data = state.get("raw_message", data)
                            self.reply(
                                b"* 1 FETCH (UID 42 BODY[] {" + str(len(data)).encode() + b"}"
                            )
                            self.wfile.write(data + b")\r\n")
                            self.wfile.flush()
                    elif command == b"LOGOUT":
                        self.reply(b"* BYE")
                        self.reply(tag + b" OK logout")
                        return
                    self.reply(tag + b" OK completed")

    servers = [Server(("127.0.0.1", 0), Handler) for _ in range(2)]
    for index, server in enumerate(servers):
        server.smtp = bool(index)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    settings = {
        "host": "127.0.0.1",
        "imap_port": servers[0].server_address[1],
        "smtp_port": servers[1].server_address[1],
        "imap_security": "starttls" if starttls else "tls",
        "smtp_security": "starttls" if starttls else "tls",
        "username": "test@example.test",
        "sender": "test@example.test",
        "password": "fixture-only-password",
        "certificate": cert.read_text(),
        "send_ledger": str(tmp_path / "sends.sqlite3"),
    }
    try:
        yield settings, state
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
