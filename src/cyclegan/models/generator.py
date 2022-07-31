import torch
import torch.nn as nn
from collections import deque

from .base_model import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(BaseModel):
    """
    The generator inputs a random noise Z, and generates an image based on Z.

    Input:
        - Z: A random noise.
    Output:
        - An image that is generated from Z.
    """

    def __init__(self, feature_num: int = 64, z_dim: int = 3):
        super().__init__()

        # We keep the last image generated to calculate losses
        self.last_generated = deque(maxlen=100)
        self.last_generated.append(torch.full((256, 256), 1))

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=feature_num, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(num_features=feature_num),
            nn.ReLU(True),

            nn.Conv2d(in_channels=feature_num, out_channels=2 * feature_num, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=2 * feature_num),
            nn.ReLU(True),

            nn.Conv2d(in_channels=2 * feature_num, out_channels=4 * feature_num, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(num_features=4 * feature_num),
            nn.ReLU(True),

            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),
            ResidualBlock(4 * feature_num),

            nn.ConvTranspose2d(in_channels=4 * feature_num, out_channels=8 * feature_num, kernel_size=3, stride=1,
                               padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(num_features=2 * feature_num),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=2 * feature_num, out_channels=4 * feature_num, kernel_size=3, stride=1,
                               padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(num_features=feature_num),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=feature_num, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.layers(z)
        self.last_generated.append(output)
        return output
