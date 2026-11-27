from types import SimpleNamespace

import pytest

from app.services import mcp_oauth_store


@pytest.mark.parametrize("operation", ["read", "write", "delete", "lock"])
def test_windows_rejects_credential_operations_before_accessing_files(monkeypatch, operation):
    monkeypatch.setattr(mcp_oauth_store, "sys", SimpleNamespace(platform="win32"))

    def unexpected_directory_access():
        pytest.fail("Unsupported platforms must not access the credential directory")

    monkeypatch.setattr(mcp_oauth_store, "default_data_dir", unexpected_directory_access)
    credential_id = "a" * 32
    with pytest.raises(mcp_oauth_store.CredentialStoreError, match="requires macOS or Linux"):
        if operation == "read":
            mcp_oauth_store.read_credentials(credential_id)
        elif operation == "write":
            mcp_oauth_store.write_credentials(credential_id, {})
        elif operation == "delete":
            mcp_oauth_store.delete_credentials(credential_id)
        else:
            with mcp_oauth_store.connection_lock(1):
                pytest.fail("Unsupported platforms must not acquire a lock")
