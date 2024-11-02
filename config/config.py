import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'lr_period': 2,
    'lr_decay': 0.9
}
