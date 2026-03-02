import torch

def _collate_fn(batches):
    if len(batches) == 0:
        return batches
    # The actual fix:
    if isinstance(batches[0], (tuple, list)) and len(batches[0]) > 0:
        if isinstance(batches[0][0], dict):
            print("Received normal batch, Size:", len(batches))
        elif isinstance(batches[0][0], (tuple, list)) and isinstance(batches[0][0][0], dict):
            print("Received NESTED LIST! Applying fix... Size:", len(batches))
            batches = [b for batch in batches for b in batch]
            print("Fixed batch size:", len(batches))
    return batches

print("--- Test Normal ---")
batches_normal = [({"encoder_length": 1}, (torch.tensor([1]), None))] * 4
_collate_fn(batches_normal)

print("--- Test Nested (DDP/BatchSampler issue) ---")
# PL wraps the sampler so DataLoader loops over [[batch1, batch2], [batch3, batch4]] instead of [batch1, batch2]
batches_nested = [
    [({"encoder_length": 1}, (torch.tensor([1]), None))] * 2,
]
_collate_fn(batches_nested)
