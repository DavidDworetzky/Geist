import stat

from scripts.ensure_operator_token import ensure_operator_token


def test_ensure_operator_token_creates_private_stable_token(tmp_path):
    token_path = tmp_path / "operator-token"

    ensure_operator_token(token_path)
    first = token_path.read_text(encoding="utf-8")
    ensure_operator_token(token_path)

    assert token_path.read_text(encoding="utf-8") == first
    assert first.startswith("geist_")
    assert len(first.strip()) >= 32
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
