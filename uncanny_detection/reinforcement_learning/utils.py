import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_reward(reward, mean, std):
    """Normalize the reward using the provided mean and standard deviation."""
    return (reward - mean) / (std + 1e-8)

def denormalize_reward(normalized_reward, mean, std):
    """Denormalize the reward using the provided mean and standard deviation."""
    return (normalized_reward * (std + 1e-8)) + mean