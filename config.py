import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,
    'valid_ratio': 0.2,
    'n_epochs': 15,
    'batch_size': 256,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'lr_period': 2,
    'lr_decay': 0.9,
    'save_path': './model.ckpt'
}