import json
import ssl
import sys
from pathlib import Path

import pytest

from app.services.email_mcp import READ_TOOLS, EmailError, MailClient
from app.services.mcp_client import McpConnection, McpError, McpServerConfig
from tests.fixtures.mail_bridge import bridge_fixture


@pytest.mark.parametrize("starttls", [False, True])
def test_real_tls_mail_read_search_send_and_replay(tmp_path, starttls):
    with bridge_fixture(tmp_path, starttls=starttls) as (settings, state):
        client = MailClient(settings)
        assert client.test() == {"ok": True}
        assert "INBOX" in str(client.call("list_folders", {}))
        found = client.call("search_messages", {"query": "hello", "limit": 1})
        assert found["uids"] == [42]
        message = client.call("read_message", {"uid": 42, "uid_validity": found["uid_validity"]})
        assert message["subject"] == "Fixture message"
        assert message["body"] == "Hello from local fixture."
        assert b"EXAMINE" in state["commands"]
        with pytest.raises(EmailError, match="identity changed"):
            client.call("read_message", {"uid": 42, "uid_validity": 999})
        args = {"to": ["recipient@example.test"], "subject": "Test", "body": "Hello"}
        assert client.call("send_message", args, "a" * 64)["status"] == "sent"
        assert MailClient(settings).call("send_message", args, "a" * 64)["replayed"]
        assert len(state["messages"]) == 1
        assert b"To: recipient@example.test" in state["messages"][0]


def test_uncertain_delivery_is_not_retried_after_restart(tmp_path):
    with bridge_fixture(tmp_path) as (settings, state):
        state["disconnect_send"] = True
        args = {"to": ["recipient@example.test"], "subject": "Test", "body": "Hello"}
        with pytest.raises(EmailError, match="uncertain"):
            MailClient(settings).call("send_message", args, "b" * 64)
        state["disconnect_send"] = False
        with pytest.raises(EmailError, match="uncertain"):
            MailClient(settings).call("send_message", args, "b" * 64)
        assert len(state["messages"]) == 1


def test_untrusted_certificate_and_remote_host_rejected(tmp_path):
    with bridge_fixture(tmp_path) as (settings, _):
        with pytest.raises(EmailError):
            MailClient({**settings, "host": "192.0.2.1"})
        other = tmp_path / "other"
        other.mkdir()
        with (
            bridge_fixture(other) as (other_settings, _),
            pytest.raises(ssl.SSLCertVerificationError),
        ):
            MailClient({**settings, "certificate": other_settings["certificate"]}).test()
        with pytest.raises(EmailError):
            MailClient({**settings, "certificate": "PRIVATE KEY"})


def test_real_stdio_mcp_uses_private_initialization_and_bounded_tools(tmp_path, monkeypatch):
    with bridge_fixture(tmp_path) as (settings, state):
        from app.services import email_credentials

        monkeypatch.setattr(
            email_credentials, "credentials", lambda reference: "fixture-only-password"
        )
        managed = {k: v for k, v in settings.items() if k != "password"}
        managed["credential_id"] = "a" * 32
        config = McpServerConfig(
            server_id=1,
            name="proton",
            transport="stdio",
            command=sys.executable,
            args=(str(Path("app/services/email_mcp.py").resolve()),),
            managed_email=managed,
        )
        assert "fixture-only-password" not in repr(config)
        connection = McpConnection(config)
        try:
            assert {tool["name"] for tool in connection.list_tools()} == READ_TOOLS | {
                "send_message"
            }
            assert json.loads(connection.call_tool("search_messages", {}))["uids"] == [42]
            assert (
                json.loads(connection.call_tool("read_message", {"uid": 42, "uid_validity": 123}))[
                    "subject"
                ]
                == "Fixture message"
            )
            with pytest.raises(McpError, match="Invalid subject"):
                connection.call_tool(
                    "send_message",
                    {
                        "to": ["recipient@example.test"],
                        "subject": "Hello\r\nBcc: attacker@example.test",
                        "body": "test",
                    },
                    idempotency_key="c" * 64,
                )
            with pytest.raises(McpError, match="control characters"):
                connection.call_tool(
                    "search_messages", {"query": "a\r\nSTORE 1 +FLAGS (\\Deleted)"}
                )
            assert state["messages"] == []
        finally:
            connection.close()


def test_unicode_message_survives_stdio_result_limit(tmp_path, monkeypatch):
    with bridge_fixture(tmp_path) as (settings, state):
        state["raw_message"] = (
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n" + ("字" * 20000).encode()
        )
        monkeypatch.setattr(
            "app.services.email_credentials.credentials", lambda reference: settings["password"]
        )
        managed = {k: v for k, v in settings.items() if k != "password"}
        managed["credential_id"] = "a" * 32
        connection = McpConnection(
            McpServerConfig(
                server_id=1,
                name="unicode",
                transport="stdio",
                command=sys.executable,
                args=(str(Path("app/services/email_mcp.py").resolve()),),
                managed_email=managed,
            )
        )
        try:
            result = json.loads(
                connection.call_tool("read_message", {"uid": 42, "uid_validity": 123})
            )
            assert result["body"] == "字" * 20000
        finally:
            connection.close()
