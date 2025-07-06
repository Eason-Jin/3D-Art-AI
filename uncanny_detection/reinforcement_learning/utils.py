import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INITIAL_THRESHOLDS = [0.4, 0.3]
