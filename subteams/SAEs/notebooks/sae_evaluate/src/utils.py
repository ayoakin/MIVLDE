import random
import numpy as np
np.infty = np.inf

import torch

def set_random_seed(seed: int, cudnn_deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch (both CPU and CUDA).
    Also configures PyTorch's CuDNN for reproducible results if requested.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
