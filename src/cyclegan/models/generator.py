import torch
import torch.nn as nn
from collections import deque

from .base_model import BaseModel


class Generator(BaseModel):
    """
    The generator inputs a random noise Z, and generates an image based on Z.

    Input:
        - Z: A random noise.
    Output:
        - An image that is generated from Z.
    """

    def __init__(self, z_dim: int = 3):
        super().__init__()

        # We keep the last image generated to calculate losses
        self.last_generated = deque(maxlen=100)
        self.last_generated.append(torch.full((256, 256), 1))

        self.layers = nn.Sequential(
            # Input of network is a random z
            nn.Conv2d(in_channels=z_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),

            # Starting to decrease depth back to 3 channels
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Output should be an RGB image with 3 channels
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.layers(z)
        self.last_generated.append(output)
        return output
