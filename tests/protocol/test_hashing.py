import pytest

from protocol.hashing import MerkleTree, canonical_hash


def leaves(n):
    return [canonical_hash({"i": i}) for i in range(n)]


def test_canonical_hash_is_order_insensitive_for_keys():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_canonical_hash_differs_for_different_payloads():
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_merkle_proofs_verify_for_every_leaf(n):
    leaf_hashes = leaves(n)
    tree = MerkleTree(leaf_hashes)
    for index, leaf in enumerate(leaf_hashes):
        proof = tree.proof(index)
        assert MerkleTree.verify(leaf, proof, tree.root)


def test_merkle_proof_rejects_tampered_leaf():
    leaf_hashes = leaves(5)
    tree = MerkleTree(leaf_hashes)
    proof = tree.proof(2)
    forged = canonical_hash({"i": "forged"})
    assert not MerkleTree.verify(forged, proof, tree.root)


def test_merkle_root_changes_when_any_leaf_changes():
    original = MerkleTree(leaves(6)).root
    altered = leaves(6)
    altered[3] = canonical_hash({"i": "changed"})
    assert MerkleTree(altered).root != original


def test_merkle_tree_requires_leaves():
    with pytest.raises(ValueError):
        MerkleTree([])
