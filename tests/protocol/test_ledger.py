import pytest

from protocol.ledger import Ledger, LedgerError


def test_escrow_lock_release_and_refund():
    ledger = Ledger()
    ledger.deposit("client", 100)
    ledger.lock_escrow("block-1", "client", 60)
    assert ledger.account("client").balance == 40

    ledger.release_escrow_to("block-1", "provider")
    assert ledger.account("provider").balance == 60
    with pytest.raises(LedgerError):
        ledger.refund_escrow("block-1")


def test_escrow_requires_sufficient_balance():
    ledger = Ledger()
    ledger.deposit("client", 10)
    with pytest.raises(LedgerError):
        ledger.lock_escrow("block-1", "client", 60)


def test_stake_and_slash():
    ledger = Ledger()
    ledger.deposit("provider", 1_000)
    ledger.add_stake("provider", 800)
    assert ledger.account("provider").balance == 200

    slashed = ledger.slash("provider", 0.5, "client")
    assert slashed == 400
    assert ledger.account("provider").stake == 400
    assert ledger.account("client").balance == 400


def test_stake_requires_balance():
    ledger = Ledger()
    with pytest.raises(LedgerError):
        ledger.add_stake("provider", 100)
