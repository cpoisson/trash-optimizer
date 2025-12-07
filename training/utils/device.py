"""Device detection utilities."""
import torch


def get_device():
    """Get the available device (CUDA, MPS, or CPU) for PyTorch operations.

    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
