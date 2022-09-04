import torch

from cyclegan.models.cycle_gan import Generator, Discriminator
from torchviz import make_dot, make_dot_from_trace

if __name__ == '__main__':
    generator = Generator()
    discriminator = Discriminator()

    print("Generator Architecture:")
    print(generator)

    print("\nDiscriminator Architecture:")
    print(discriminator)

    x = torch.randn(3, 256, 256)
    make_dot(generator(x), params=dict(generator.named_parameters()))
    make_dot(discriminator(x), params=dict(discriminator.named_parameters()))
