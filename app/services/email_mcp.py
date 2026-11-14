"""Bundled, local-only IMAP/SMTP MCP server. Stdout is exclusively JSON-RPC."""

from __future__ import annotations

import contextlib
import email.policy
import imaplib
import ipaddress
import json
import re
import smtplib
import sqlite3
import ssl
import sys
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import make_msgid
from pathlib import Path
from typing import Any, cast


MAX_MESSAGE_BYTES = 512_000
READ_TOOLS = frozenset({"list_folders", "search_messages", "read_message"})


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


FOLDER = {"type": "string", "maxLength": 512, "default": "INBOX"}
TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_folders",
        "description": "List mail folders without changing messages.",
        "inputSchema": _schema({}),
    },
    {
        "name": "search_messages",
        "description": "Search a folder by literal text. Returns UIDs and UID validity for read_message. Does not mark messages read.",
        "inputSchema": _schema(
            {
                "folder": FOLDER,
                "query": {"type": "string", "maxLength": 500},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }
        ),
    },
    {
        "name": "read_message",
        "description": "Read a message by UID and UID validity without marking it read. Attachments are metadata only.",
        "inputSchema": _schema(
            {
                "folder": FOLDER,
                "uid": {"type": "integer", "minimum": 1},
                "uid_validity": {"type": "integer", "minimum": 1},
            },
            ["uid", "uid_validity"],
        ),
    },
    {
        "name": "send_message",
        "description": "Send a plain-text email from the connected account. Requires approval. Never retry an uncertain send automatically.",
        "inputSchema": _schema(
            {
                "to": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 254},
                    "minItems": 1,
                    "maxItems": 20,
                },
                "subject": {"type": "string", "maxLength": 998},
                "body": {"type": "string", "maxLength": 20000},
            },
            ["to", "subject", "body"],
        ),
    },
]


class EmailError(ValueError):
    pass


def validate_settings(settings: dict) -> None:
    try:
        if not ipaddress.ip_address(settings["host"]).is_loopback:
            raise ValueError()
        for field in ("imap_port", "smtp_port"):
            if type(settings[field]) is not int or not 1 <= settings[field] <= 65535:
                raise ValueError()
        for field in ("imap_security", "smtp_security"):
            if settings[field] not in ("starttls", "tls"):
                raise ValueError()
        for field in ("username", "sender"):
            if (
                not isinstance(settings[field], str)
                or not settings[field]
                or len(settings[field]) > 254
                or re.search(r"[\r\n\x00]", settings[field])
            ):
                raise ValueError()
        validate_address(settings["sender"])
        if not settings.get("password") or len(settings["password"]) > 4096:
            raise ValueError()
        cert = settings["certificate"]
        if len(cert) > 16000 or "PRIVATE KEY" in cert or "-----BEGIN CERTIFICATE-----" not in cert:
            raise ValueError()
        tls_context(settings)
    except (KeyError, TypeError, ValueError, ssl.SSLError) as error:
        raise EmailError(
            "Invalid Bridge settings: use a loopback IP, valid ports, TLS/STARTTLS, and the exported public certificate"
        ) from error


def tls_context(settings: dict) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    # Trust only the explicitly supplied Bridge certificate. Bridge certificates
    # need not carry a loopback IP SAN; certificate-chain verification stays on.
    context.check_hostname = False
    context.load_verify_locations(cadata=settings["certificate"])
    return context


def validate_address(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) > 254
        or not re.fullmatch(
            r"[^\s<>@,;\x00-\x1f]+@[^\s<>@,;\x00-\x1f]+\.[^\s<>@,;\x00-\x1f]+", value
        )
    ):
        raise EmailError("Use plain email addresses without display names")


def _quoted(value: str) -> str:
    if not isinstance(value, str) or re.search(r"[\x00-\x1f\x7f]", value):
        raise EmailError("Folder and search text must not contain control characters")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ok(status: str, data: Any) -> Any:
    if status != "OK":
        raise EmailError("Bridge could not complete the mailbox operation")
    return data


class MailClient:
    def __init__(self, settings: dict):
        validate_settings(settings)
        self.settings = settings

    @contextlib.contextmanager
    def imap(self):
        s = self.settings
        client: imaplib.IMAP4 | None = None
        try:
            if s["imap_security"] == "tls":
                client = imaplib.IMAP4_SSL(
                    s["host"], s["imap_port"], ssl_context=tls_context(s), timeout=10
                )
            else:
                client = imaplib.IMAP4(s["host"], s["imap_port"], timeout=10)
                client.starttls(ssl_context=tls_context(s))
            client.login(s["username"], s["password"])
            yield client
        finally:
            if client:
                with contextlib.suppress(Exception):
                    client.logout()

    @contextlib.contextmanager
    def smtp(self):
        s = self.settings
        client: smtplib.SMTP | None = None
        try:
            if s["smtp_security"] == "tls":
                client = smtplib.SMTP_SSL(
                    s["host"], s["smtp_port"], context=tls_context(s), timeout=10
                )
            else:
                client = smtplib.SMTP(s["host"], s["smtp_port"], timeout=10)
                client.ehlo()
                client.starttls(context=tls_context(s))
            client.ehlo()
            client.login(s["username"], s["password"])
            yield client
        finally:
            if client:
                with contextlib.suppress(Exception):
                    client.quit()
                client.close()

    def test(self) -> dict:
        with self.imap() as client:
            _ok(*client.noop())
        with self.smtp() as client:
            code, _ = client.noop()
            if code != 250:
                raise EmailError("SMTP authentication test failed")
        return {"ok": True}

    def call(self, name: str, arguments: dict, idempotency_key: str | None = None) -> dict:
        tool = next((tool for tool in TOOLS if tool["name"] == name), None)
        if tool is None or not isinstance(arguments, dict):
            raise EmailError("Unknown email tool")
        schema = tool["inputSchema"]
        if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(arguments):
            raise EmailError("Invalid email tool arguments")
        if name == "send_message":
            return self.send(arguments, idempotency_key)
        folder = arguments.get("folder", "INBOX")
        if not isinstance(folder, str) or len(folder) > 512:
            raise EmailError("Invalid folder")
        with self.imap() as client:
            if name == "list_folders":
                rows = _ok(*client.list()) or []
                return {
                    "folders": [
                        row.decode("utf-8", errors="replace")[:1024]
                        for row in rows[:100]
                        if isinstance(row, bytes)
                    ],
                    "truncated": len(rows) > 100,
                }
            _ok(*client.select(_quoted(folder), readonly=True))
            _, validity = client.response("UIDVALIDITY")
            if not validity or not validity[0]:
                raise EmailError("Bridge did not provide stable mailbox identity")
            uid_validity = int(validity[0])
            if name == "search_messages":
                query, limit = arguments.get("query", ""), arguments.get("limit", 20)
                if (
                    not isinstance(query, str)
                    or len(query) > 500
                    or type(limit) is not int
                    or not 1 <= limit <= 50
                ):
                    raise EmailError("Invalid search text or limit")
                # IMAP literals are not needed for the bounded ASCII search surface.
                # Reject unsupported encoding rather than interpolate raw IMAP syntax.
                if not query.isascii():
                    raise EmailError("Search currently supports ASCII text only")
                criteria = ("TEXT", _quoted(query)) if query else ("ALL",)
                rows = _ok(*client.uid("search", None, *criteria))
                uids = (rows[0] or b"").split()
                return {
                    "folder": folder,
                    "uid_validity": uid_validity,
                    "uids": [int(uid) for uid in uids[-limit:][::-1]],
                    "truncated": len(uids) > limit,
                }
            uid = arguments["uid"]
            if (
                type(uid) is not int
                or not 1 <= uid <= 4294967295
                or type(arguments["uid_validity"]) is not int
            ):
                raise EmailError("Invalid message UID")
            if arguments["uid_validity"] != uid_validity:
                raise EmailError("Mailbox identity changed; search again before reading")
            rows = _ok(*client.uid("fetch", str(uid), f"(BODY.PEEK[]<0.{MAX_MESSAGE_BYTES + 1}>)"))
            raw = next((row[1] for row in rows if isinstance(row, tuple)), None)
            if raw is None:
                raise EmailError("Message not found")
            if len(raw) > MAX_MESSAGE_BYTES:
                raise EmailError("Message exceeds the supported 512 KB reading limit")
            message = cast(EmailMessage, BytesParser(policy=email.policy.default).parsebytes(raw))
            part = message.get_body(preferencelist=("plain", "html"))
            body = part.get_content() if part else ""
            attachments = [
                {
                    "filename": attachment.get_filename(),
                    "content_type": attachment.get_content_type(),
                }
                for attachment in list(message.iter_attachments())[:50]
            ]
            return {
                "folder": folder,
                "uid": uid,
                "uid_validity": uid_validity,
                "from": str(message.get("From", ""))[:2000],
                "to": str(message.get("To", ""))[:2000],
                "subject": str(message.get("Subject", ""))[:2000],
                "date": str(message.get("Date", ""))[:200],
                "body": str(body)[:20000],
                "truncated": len(str(body)) > 20000,
                "attachments": attachments,
            }

    def send(self, arguments: dict, key: str | None) -> dict:
        recipients, subject, body = arguments["to"], arguments["subject"], arguments["body"]
        if not isinstance(recipients, list) or not 1 <= len(recipients) <= 20:
            raise EmailError("Provide 1 to 20 recipients")
        for address in recipients:
            validate_address(address)
        if (
            not isinstance(subject, str)
            or len(subject) > 998
            or re.search(r"[\r\n\x00]", subject)
            or not isinstance(body, str)
            or len(body) > 20000
        ):
            raise EmailError("Invalid subject or body")
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise EmailError("Sending requires a Geist idempotency key")
        message = EmailMessage()
        message["From"] = self.settings["sender"]
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        message["Message-ID"] = make_msgid()
        message.set_content(body)
        # Only a runtime-owned path supplied at initialization can select this DB.
        path = Path(self.settings["send_ledger"])
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.closing(sqlite3.connect(path, timeout=10)) as ledger:
            path.chmod(0o600)
            ledger.execute(
                "CREATE TABLE IF NOT EXISTS sends (key TEXT PRIMARY KEY, status TEXT, message_id TEXT)"
            )
            prior = ledger.execute(
                "SELECT status, message_id FROM sends WHERE key = ?", (key,)
            ).fetchone()
            if prior:
                if prior[0] == "sent":
                    return {"status": "sent", "message_id": prior[1], "replayed": True}
                raise EmailError(
                    "Delivery is uncertain or partial; check Sent mail before composing another send"
                )
            with self.smtp() as client:
                try:
                    ledger.execute(
                        "INSERT INTO sends VALUES (?, ?, ?)",
                        (key, "attempted", str(message["Message-ID"])),
                    )
                    ledger.commit()
                except sqlite3.IntegrityError as error:
                    raise EmailError("This send is already in progress; check Sent mail") from error
                try:
                    refused = client.send_message(
                        message, from_addr=self.settings["sender"], to_addrs=recipients
                    )
                except (OSError, smtplib.SMTPException) as error:
                    raise EmailError(
                        "Delivery is uncertain; check Sent mail before composing another send"
                    ) from error
                if refused:
                    raise EmailError(
                        "Some recipients were refused; check Sent mail before sending again"
                    )
                ledger.execute("UPDATE sends SET status = ? WHERE key = ?", ("sent", key))
                ledger.commit()
        return {"status": "sent", "message_id": str(message["Message-ID"])}


def main() -> None:
    client = None
    result: dict[str, Any]
    while line := sys.stdin.buffer.readline(128_001):
        if len(line) > 128_000:
            return
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or "id" not in request:
                continue
            method, params = request.get("method"), request.get("params") or {}
            if method == "initialize":
                if client is not None:
                    raise EmailError("Already initialized")
                client = MailClient(params["geistEmail"])
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "geist-email", "version": "1.0"},
                }
            elif client is None:
                raise EmailError("Initialize first")
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    value = client.call(
                        params.get("name", ""),
                        params.get("arguments", {}),
                        (params.get("_meta") or {}).get("geist/idempotencyKey"),
                    )
                    result = {
                        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]
                    }
                except EmailError as error:
                    result = {"isError": True, "content": [{"type": "text", "text": str(error)}]}
                except Exception:
                    result = {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": "Bridge operation failed. Check Bridge, TLS certificate, and credentials.",
                            }
                        ],
                    }
            elif method == "ping":
                result = {}
            else:
                raise EmailError("Unsupported method")
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32602, "message": "Invalid email MCP request or configuration"},
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
