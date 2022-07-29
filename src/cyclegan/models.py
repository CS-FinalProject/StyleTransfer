import torch.nn as nn

import torch.nn as nn
# import torch.nn.functional as F
import torch


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


class Generator(nn.Module):
    """
    The generator inputs a random noise Z, and generates an image based on Z.

    Input:
       - Z: A random noise.
    Output:
       - An image that is generated from Z.
    """

    def __init__(self, feacher_num: int = 64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=feacher_num, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(num_features=feacher_num),
            nn.ReLU(True),

            nn.Conv2d(in_channels=feacher_num, out_channels=2 * feacher_num, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=2 * feacher_num),
            nn.ReLU(True),

            nn.Conv2d(in_channels=2 * feacher_num, out_channels=4 * feacher_num, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=4 * feacher_num),
            nn.ReLU(True),

            ResidualBlock(4 * feacher_num),
            ResidualBlock(4 * feacher_num),
            ResidualBlock(4 * feacher_num),
            ResidualBlock(4 * feacher_num),

            nn.ConvTranspose2d(in_channels=4 * feacher_num, out_channels=8 * feacher_num, kernel_size=3, stride=1,
                               padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(num_features=2 * feacher_num),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=2 * feacher_num, out_channels=4 * feacher_num, kernel_size=3, stride=1,
                               padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(num_features=feacher_num),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=feacher_num, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

        # def __init__(self, z_dim: int = 3):
        #
        #     super().__init__()
        #
        #     self.layers = nn.Sequential(
        #         # Input of network is a random z
        #         nn.ConvTranspose2d(in_channels=z_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(True),
        #
        #         nn.ConvTranspose2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(96),
        #         nn.ReLU(True),
        #
        #         nn.ConvTranspose2d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(True),
        #
        #         nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(True),
        #
        #         # Output should be an RGB image with 3 channels
        #         nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
        #         nn.Tanh()
        #     )
        #
        #     # self.layers = nn.Sequential(
        #     #     # Input is an RGB image
        #     #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
        #     #     nn.BatchNorm2d(32),
        #     #     nn.ReLU(),
        #     #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        #     #     nn.BatchNorm2d(64),
        #     #     nn.ReLU(),
        #     #     nn.MaxPool2d(2),
        #     #     nn.BatchNorm2d(64),
        #     #     nn.ReLU(),
        #     #     nn.Dropout(0.3),
        #     #
        #     #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        #     #     nn.BatchNorm2d(128),
        #     #     nn.ReLU(),
        #     #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        #     #     nn.BatchNorm2d(128),
        #     #     nn.ReLU(),
        #     #     nn.MaxPool2d(2),
        #     #     nn.BatchNorm2d(64),
        #     #     nn.ReLU(),
        #     #     nn.Dropout(0.3),
        #     #
        #     #     # One output (real/fake)
        #     #     nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1)
        #     # )

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    """
    The discriminator predicts whether an image is real or fake, when the image is generated from the Generator.

    Input:
        - image: An image whose either real or generated by the Generator.
    Output:
        - A single value, representing the probability that the image is real.
    """

    def __init__(self, in_feacher: int = 3, out_feacher: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_feacher, out_channels=out_feacher, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=out_feacher, out_channels=2 * out_feacher, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(2 * out_feacher),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=2 * out_feacher, out_channels=4 * out_feacher, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * out_feacher),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=4 * out_feacher, out_channels=8 * out_feacher, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(8 * out_feacher),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=8 * out_feacher, out_channels=1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    # self.layers = nn.Sequential(
    #         # Input is an RGB image
    #         nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
    #         nn.LeakyReLU(0.2, inplace=True),
    #
    #         nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(128),
    #         nn.LeakyReLU(0.2, inplace=True),
    #
    #         nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(256),
    #         nn.LeakyReLU(0.2, inplace=True),
    #
    #         nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
    #         nn.BatchNorm2d(512),
    #         nn.LeakyReLU(0.2, inplace=True),
    #
    #         # Output is a single value, representing the probability that the image is real.
    #         nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=1),
    #         nn.Tanh()
    #     )
    #
    #     # self.layers = nn.Sequential(
    #     #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
    #     #     nn.BatchNorm2d(32),
    #     #     nn.ReLU(),
    #     #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    #     #     nn.BatchNorm2d(64),
    #     #     nn.ReLU(),
    #     #     nn.MaxPool2d(2),
    #     #     nn.BatchNorm2d(64),
    #     #     nn.ReLU(),
    #     #     nn.Dropout(0.3),
    #     #
    #     #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
    #     #     nn.BatchNorm2d(128),
    #     #     nn.ReLU(),
    #     #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
    #     #     nn.BatchNorm2d(128),
    #     #     nn.ReLU(),
    #     #     nn.MaxPool2d(2),
    #     #     nn.BatchNorm2d(64),
    #     #     nn.ReLU(),
    #     #     nn.Dropout(0.3),
    #     #
    #     #     # One output (real/fake)
    #     #     nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
    #     # )

    def forward(self, image):
        """
        Forward pass of the discriminator.
        """
        return self.layers(image)
