import argparse

import torch

from cyclegan_pytorch import ImageDataset
from cyclegan_pytorch import Generator, Discriminator
import torch.nn as nn
import logging


def train_model(dataset_path: str, epochs: int, batch_size: int, learning_rate: float, generator_A2B: Generator,
                generator_B2A: Generator, discriminator_a: Discriminator, discriminator_b: Discriminator,
                cycle_loss_func: nn.modules.loss, adversarial_loss_func: nn.modules.loss, optimizer_D_A, optimizer_D_B,
                optimizer_G, device):
    dataset = ImageDataset(dataset_path)

    for epoch_idx in range(epochs):
        for items in dataset:
            item_a, item_b = items["A"].to(device), items["B"].to(device)

            # Cycle GAN Loss
            generated_image_B = generator_A2B(item_a)
            generated_item_a = generator_B2A(generated_image_B)

            generated_image_A = generator_B2A(item_b)
            generated_item_b = generator_A2B(generated_image_A)

            cycle_gan_loss = cycle_loss_func(generated_item_a - item_a) + cycle_loss_func(generated_item_b - item_b)

            # Adversarial loss
            adversarial_loss_b = adversarial_loss_func(
                torch.log(discriminator_b(item_b)) + torch.log(1 - discriminator_b(generated_item_b)))

            adversarial_loss_a = adversarial_loss_func(
                torch.log(discriminator_a(item_a)) + torch.log(1 - discriminator_a(generated_item_a)))

            total_loss = cycle_gan_loss + adversarial_loss_a + adversarial_loss_b

            # Update generators
            total_loss.backward()
            optimizer_G.step()

            # Update discriminators
            discriminator_a_loss = update_discriminator(discriminator_a, item_a, generated_item_a,
                                                        adversarial_loss_func, batch_size, device,
                                                        optimizer_D_A)
            discriminator_b_loss = update_discriminator(discriminator_b, item_b, generated_item_b,
                                                        adversarial_loss_func, batch_size, device,
                                                        optimizer_D_B)

            logging.info("[{}/{}] | G loss: {} | D loss: {}".format(epoch_idx, epochs, total_loss,
                                                                    discriminator_b_loss + discriminator_a_loss))


def update_discriminator(discriminator: Discriminator, real_image, fake_image, loss_func, batch_size: int, device,
                         optimizer) -> float:
    """
    Updates loss of discriminator neural net, according to its loss.
    :param optimizer:
    :param discriminator:
    :param real_image:
    :param fake_image:
    :param loss_func:
    :param batch_size:
    :param device:
    :return:
    """
    real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
    fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

    output_real_image = discriminator(real_image)
    output_fake_image = discriminator(fake_image)

    loss = loss_func(real_label - output_real_image) + loss_func(fake_label - output_fake_image)
    loss.backwards()
    optimizer.step()

    return loss


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset", help="Path to the monet2photo dataset", default="./monet2photo")
    arg_parser.add_argument("image-size", help="Size of data crop", default=256)

    G_A2B = Generator()
    G_B2A = Generator()

    D_A = Discriminator()
    D_B = Discriminator()






if __name__ == '__main__':
    main()
