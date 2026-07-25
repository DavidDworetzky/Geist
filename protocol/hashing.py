"""Canonical hashing and Merkle commitments for the compute protocol."""

import hashlib
import json
from dataclasses import dataclass


# Domain separation prevents second-preimage attacks where an interior node is
# presented as a leaf.
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def canonical_json(payload: dict | list) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(payload: dict | list) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(_LEAF_PREFIX + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


@dataclass(frozen=True)
class MerkleProof:
    """Inclusion proof for one leaf: sibling hashes from leaf to root."""

    leaf_index: int
    siblings: list[tuple[str, bool]]  # (hex hash, sibling_is_left)


class MerkleTree:
    """Binary Merkle tree over an ordered list of hex-encoded leaf hashes."""

    def __init__(self, leaf_hashes: list[str]):
        if not leaf_hashes:
            raise ValueError("Merkle tree requires at least one leaf")
        self._levels: list[list[bytes]] = [
            [_leaf_hash(bytes.fromhex(leaf)) for leaf in leaf_hashes]
        ]
        while len(self._levels[-1]) > 1:
            level = self._levels[-1]
            if len(level) % 2 == 1:
                level = level + [level[-1]]
            self._levels.append(
                [_node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
            )

    @property
    def root(self) -> str:
        return self._levels[-1][0].hex()

    def proof(self, leaf_index: int) -> MerkleProof:
        if not 0 <= leaf_index < len(self._levels[0]):
            raise IndexError(f"leaf index {leaf_index} out of range")
        siblings: list[tuple[str, bool]] = []
        index = leaf_index
        for level in self._levels[:-1]:
            padded = level + [level[-1]] if len(level) % 2 == 1 else level
            sibling_index = index - 1 if index % 2 == 1 else index + 1
            siblings.append((padded[sibling_index].hex(), sibling_index < index))
            index //= 2
        return MerkleProof(leaf_index=leaf_index, siblings=siblings)

    @staticmethod
    def verify(leaf_hash_hex: str, proof: MerkleProof, root: str) -> bool:
        node = _leaf_hash(bytes.fromhex(leaf_hash_hex))
        for sibling_hex, sibling_is_left in proof.siblings:
            sibling = bytes.fromhex(sibling_hex)
            node = _node_hash(sibling, node) if sibling_is_left else _node_hash(node, sibling)
        return node.hex() == root
