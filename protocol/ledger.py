"""In-memory accounting: balances, provider stake, and per-block escrow.

This models the settlement layer of the protocol. In a deployed network the
same transitions would live on a chain or a trusted settlement service; the
coordinator only depends on this interface.
"""

from dataclasses import dataclass, field


class LedgerError(Exception):
    pass


@dataclass
class Account:
    balance: int = 0
    stake: int = 0


@dataclass
class Escrow:
    client_id: str
    amount: int
    released: bool = False


@dataclass
class Ledger:
    accounts: dict[str, Account] = field(default_factory=dict)
    escrows: dict[str, Escrow] = field(default_factory=dict)

    def account(self, account_id: str) -> Account:
        return self.accounts.setdefault(account_id, Account())

    def deposit(self, account_id: str, amount: int) -> None:
        self._require_positive(amount)
        self.account(account_id).balance += amount

    def add_stake(self, account_id: str, amount: int) -> None:
        self._require_positive(amount)
        account = self.account(account_id)
        if account.balance < amount:
            raise LedgerError(f"{account_id} has insufficient balance to stake {amount}")
        account.balance -= amount
        account.stake += amount

    def lock_escrow(self, block_id: str, client_id: str, amount: int) -> None:
        self._require_positive(amount)
        if block_id in self.escrows:
            raise LedgerError(f"escrow already exists for block {block_id}")
        account = self.account(client_id)
        if account.balance < amount:
            raise LedgerError(f"{client_id} has insufficient balance to escrow {amount}")
        account.balance -= amount
        self.escrows[block_id] = Escrow(client_id=client_id, amount=amount)

    def release_escrow_to(self, block_id: str, recipient_id: str) -> None:
        escrow = self._open_escrow(block_id)
        escrow.released = True
        self.account(recipient_id).balance += escrow.amount

    def refund_escrow(self, block_id: str) -> None:
        escrow = self._open_escrow(block_id)
        escrow.released = True
        self.account(escrow.client_id).balance += escrow.amount

    def slash(self, provider_id: str, fraction: float, recipient_id: str) -> int:
        """Burn part of the provider's stake toward the recipient; returns the amount."""
        if not 0.0 < fraction <= 1.0:
            raise LedgerError("slash fraction must be in (0, 1]")
        account = self.account(provider_id)
        amount = int(account.stake * fraction)
        account.stake -= amount
        self.account(recipient_id).balance += amount
        return amount

    def _open_escrow(self, block_id: str) -> Escrow:
        escrow = self.escrows.get(block_id)
        if escrow is None or escrow.released:
            raise LedgerError(f"no open escrow for block {block_id}")
        return escrow

    @staticmethod
    def _require_positive(amount: int) -> None:
        if amount <= 0:
            raise LedgerError("amount must be positive")
