import torch


def build_source_ids(sources):
    """Map a list of source strings to batch-local integer ids (for rank masking)."""
    mapping = {s: i for i, s in enumerate(sorted(set(sources)))}
    return torch.tensor([mapping[s] for s in sources], dtype=torch.long)
