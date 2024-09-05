def test_mps():
    import torch

    if torch.backends.mps.is_available():
        MPS_available = True
        print("MPS available")
    else:
        MPS_available = False
        print("MPS not available")
    if torch.backends.mps.is_built():
        MPS_built = True
        print("MPS built")
    else:
        MPS_built = False
        print("MPS not built")
    if MPS_available and MPS_built:
        mps_device = torch.device("mps")
        torch.ones(5, device=mps_device)
