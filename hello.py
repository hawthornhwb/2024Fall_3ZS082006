import torch
import os

# 加载 .pth 文件
model_path = os.path.join('model_ckpt', 'fasternet_t0-epoch=281-val_acc1=71.9180.pth')  # 替换为你的文件路径
checkpoint = torch.load(model_path)
# print(type(checkpoint))

for key in checkpoint.keys():
    print(key)