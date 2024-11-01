from fasternet import FasterNet
from torch import nn
import torchvision

def finetune_fasternet_t0(device):
    pretrained_net = FasterNet(pretrained=True)
    finetune_net = nn.Sequential()
    finetune_net.features = pretrained_net

    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    finetune_net = finetune_net.to(device)

    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

def finetune_mobilenet_v2(device):
    pretrained_net = torchvision.models.mobilenet_v2(pretrained=True)
    finetune_net = nn.Sequential()
    finetune_net.features = pretrained_net

    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))

    finetune_net = finetune_net.to(device)

    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

def finetune_resnet_18(device):
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    finetune_net = nn.Sequential()
    finetune_net.features = pretrained_net

    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))

    finetune_net = finetune_net.to(device)

    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

