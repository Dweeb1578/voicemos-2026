import torch

from src.train_utils import build_source_ids


def test_build_source_ids_groups_same_strings():
    ids = build_source_ids(["bvcc", "tmhint", "bvcc"])
    assert ids[0] == ids[2]
    assert ids[0] != ids[1]
    assert ids.dtype == torch.long


def test_build_source_ids_length():
    ids = build_source_ids(["a", "b", "c", "a"])
    assert ids.shape == (4,)
