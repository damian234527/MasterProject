import random
import numpy as np
import torch
import logging
from config import GENERAL_CONFIG


def set_seed(seed_value):
    """Sets the seed on all relevant libraries for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set seed for all operations to: {seed_value}")