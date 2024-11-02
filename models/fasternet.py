import torch
from torch import nn
from typing import List


class Partial_conv3(nn.Module):
    """定义部分卷积"""

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x


class FasterNetBlock(nn.Module):
    """文献中的FasterNetBlock块，由一个部分卷积和两个点对对卷积组成"""

    def __init__(self, input_channels, num_channels, output_channels, n_div):
        super().__init__()

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, output_channels, kernel_size=1, stride=1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(input_channels, n_div)

    def forward(self, x):
        y = self.spatial_mixing(x)
        y = self.mlp(y)
        y += x
        return y


class Blocks(nn.Module):
    """FasterNet网络架构中的每一个block，每一个block由若干个FasterNetblock组成"""

    def __init__(self, embed_channels, i_stage, n_div, num_stages):
        super().__init__()
        blocks_list = [
            FasterNetBlock(
                input_channels=embed_channels * 2 ** i_stage,
                num_channels=2 * embed_channels * 2 ** i_stage,
                output_channels=embed_channels * 2 ** i_stage,
                n_div=n_div
            )
            for _ in range(num_stages[i_stage])
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class Merging(nn.Module):
    """Merging层"""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.reduction = nn.Conv2d(input_channels, output_channels, kernel_size=2, stride=2, bias=False)

        self.norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x


class Embedding(nn.Module):
    """Embedding层"""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.proj = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4, bias=False)

        self.norm = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x


class FasterNet(nn.Module):
    """FasterNet-T0网络架构"""

    def __init__(self,
                 input_channels=3,
                 embed_channels=40,
                 num_stages=[1, 2, 8, 2],
                 n_div=4,
                 feature_dim=1280,
                 num_classes=1000,
                 pretrained=False
                 ):
        super().__init__()

        self.depth = len(num_stages)
        self.num_features = int(embed_channels * 2 ** (self.depth - 1))

        self.patch_embed = Embedding(
            input_channels=input_channels,
            output_channels=embed_channels
        )

        stages_list = []

        for i_stage in range(len(num_stages)):
            stage = Blocks(
                embed_channels=embed_channels,
                i_stage=i_stage,
                n_div=n_div,
                num_stages=num_stages
            )

            stages_list.append(stage)

            if i_stage < self.depth - 1:
                patch_merged = Merging(
                    input_channels=embed_channels * 2 ** i_stage,
                    output_channels=2 * embed_channels * 2 ** i_stage,
                )
                stages_list.append(patch_merged)

        self.stages = nn.Sequential(*stages_list)

        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
            nn.GELU()
        )

        self.head = nn.Linear(feature_dim, num_classes) \
            if num_classes > 0 else nn.Identity()

        if pretrained:
            self.load_pretrained_parameters()

    def load_pretrained_parameters(self,
                                   pretrained_path='model_ckpt/fasternet_t0-epoch=281-val_acc1=71.9180.pth'):
        # 加载预训练参数
        pretrained_dict = torch.load(pretrained_path, weights_only='True')

        # 加载参数到模型
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 更新模型的状态字典
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


