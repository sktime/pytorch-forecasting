import torch
from torch.utils.data._utils.collate import default_collate

def int_target_collate(batch):
    # Handle both lists and tuples, filter out None
    try:
        filtered_batch = [item for item in list(batch) if item is not None]
    except Exception:
        filtered_batch = [] if batch is None else [batch]
    if not filtered_batch:
        # Return an empty dict to avoid collate errors
        return {}
    try:
        collated = default_collate(filtered_batch)
    except Exception as e:
        print(f"Collate failed: {e}")
        return filtered_batch
    if isinstance(collated, dict) and "target" in collated:
        collated["target"] = collated["target"].long()
    return collated

# Mock data
x1 = {"a": torch.tensor([1, 2])}
y1 = (torch.tensor([3, 4]), None)
x2 = {"a": torch.tensor([5, 6])}
y2 = (torch.tensor([7, 8]), None)

batch = [(x1, y1), (x2, y2)]
result = int_target_collate(batch)

print(f"Result type: {type(result)}")
if isinstance(result, tuple):
    print(f"Result length: {len(result)}")
    print(f"Result[0] type: {type(result[0])}")
    print(f"Result[1] type: {type(result[1])}")

x, y = result
print(f"x type: {type(x)}")
print(f"y type: {type(y)}")
