import torch
from torch import nn
import os


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
    def __init__(self, input_channels, num_channels, output_channels, n_div):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.act_layer = nn.GELU()
        self.conv2 = nn.Conv2d(num_channels, output_channels, kernel_size=1, stride=1, bias=False)
        self.pconv3 = Partial_conv3(input_channels, n_div)


    def forward(self, x):
        y = self.pconv3(x)
        y = self.act_layer(self.bn1(self.conv1(y)))
        y = self.conv2(y)
        y += x
        return y


class Merging(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=2, stride=2, bias=False)

        self.bn = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Embedding(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=4, bias=False)

        self.bn = nn.BatchNorm2d(output_channels)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class FasterNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 embed_channels=40,
                 num_stages=[1, 2, 8, 2],
                 n_div = 4,
                 feature_dim=1280,
                 num_classes=1000
                 ):
        super().__init__()

        self.depth = len(num_stages)
        self.num_features = int(embed_channels * 2 ** (self.depth - 1))

        self.patch_embed = Embedding(
            input_channels=input_channels,
            output_channels=embed_channels
        )

        # input_channels, num_channels, output_channels, n_div
        stages_list = []

        for i_stage, i_blocks in enumerate(num_stages):
            while i_blocks > 0:
                stage = FasterNetBlock(
                    input_channels=embed_channels * 2 ** (i_stage),
                    num_channels=2 * embed_channels * 2 ** (i_stage),
                    output_channels=embed_channels * 2 ** (i_stage),
                    n_div = n_div
                )
                stages_list.append(stage)
                i_blocks -= 1

            if i_stage < self.depth - 1:
                patch_merged = Merging(
                    input_channels=embed_channels * 2 ** (i_stage),
                    output_channels=2 * embed_channels * 2 ** (i_stage),
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
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x








