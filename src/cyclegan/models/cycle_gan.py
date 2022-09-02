import os

import torch.nn as nn
import torch
from os import path
from itertools import chain

import wandb
from cyclegan.utils import *

from .generator import Generator
from .discriminator import Discriminator
from .base_model import BaseModel
from .. import weights_init


class CycleGAN(BaseModel):
    def __init__(self, lr: float, lambda_param: float, continue_learning: bool, device, counter: int, decay_epoch: int):
        super().__init__()

        self.lambda_param = lambda_param
        self.device = device
        self.counter = counter
        self.decay_epoch = decay_epoch
        self.losses = dict()

        # Define the generators and discriminators
        self.generator_A2B = Generator().to(device)
        self.generator_B2A = Generator().to(device)
        self.discriminator_A = Discriminator().to(device)
        self.discriminator_B = Discriminator().to(device)
        self.init_models(continue_learning)

        # Define the loss functions
        self.identity_loss_func = torch.nn.L1Loss()
        self.adversarial_loss_func = nn.MSELoss()
        self.cycle_loss_func = torch.nn.L1Loss()

        ##############
        # Optimizers #
        ##############
        self.gen_optim = torch.optim.Adam(chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()),
                                          lr=lr, betas=(0.5, 0.999))
        self.discA_optim = torch.optim.Adam(
            self.discriminator_A.parameters(), lr=lr, betas=(0.5, 0.999))
        self.discB_optim = torch.optim.Adam(
            self.discriminator_B.parameters(), lr=lr, betas=(0.5, 0.999))

        # Defining Decay LR - Gradually changing the LR
        # self.gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optim, step_size=decay_epoch, gamma=0.1)
        # self.discA_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.discA_optim, step_size=decay_epoch, gamma=0.1)
        # self.discB_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.discB_optim, step_size=decay_epoch, gamma=0.1)

    def init_models(self, cont: bool) -> None:
        # Load the latest models if we want to continue learning
        if not cont:
            self.generator_A2B.apply(weights_init)
            self.generator_B2A.apply(weights_init)
            self.discriminator_A.apply(weights_init)
            self.discriminator_B.apply(weights_init)

    def identity_loss(self, real_image_A: torch.Tensor, real_image_B: torch.Tensor) -> tuple:
        """
        The identity loss is computed by comparing the output of the inverse generator against a real image.
        It keeps the colors form the input to the generated image.
        :return: Loss value
        """
        # Difference between identity image and real image
        loss_A = self.identity_loss_func(self.generator_B2A(real_image_A), real_image_A) * self.lambda_param / 2
        loss_B = self.identity_loss_func(self.generator_A2B(real_image_B), real_image_B) * self.lambda_param / 2

        # Track losses
        self.losses["idt_loss_A"] = loss_A
        self.losses["idt_loss_B"] = loss_B

        return loss_A, loss_B

    def adversarial_loss_generator(self, real_image_A: torch.Tensor, real_image_B: torch.Tensor,
                                   real_label: torch.Tensor) -> tuple:
        """
        The adversarial loss affects the generator and the discriminator. Its goal is to minimize the loss of the
        discriminator given an input from the generator, intuitively, to trick the generator.
        :return: Loss value
        """

        # The generator wants to "trick" the discriminator, and therefore minimize the difference
        # between the discriminator's output and the real label
        loss_A2B = self.adversarial_loss_func(self.discriminator_B(self.generator_A2B(real_image_A)), real_label)
        loss_B2A = self.adversarial_loss_func(self.discriminator_A(self.generator_B2A(real_image_B)), real_label)

        # Track losses
        self.losses["adv_loss_genA2B"] = loss_A2B
        self.losses["adv_loss_genB2A"] = loss_B2A

        return loss_A2B, loss_B2A

    def forward_cycle_loss(self, generator: Generator, inv_generator: Generator,
                           real_image: torch.Tensor) -> torch.Tensor:
        recovered_image = inv_generator(generator(real_image))
        return self.cycle_loss_func(recovered_image, real_image) * self.lambda_param

    def cycle_loss(self, realA: torch.Tensor, realB: torch.Tensor) -> tuple:
        forward_loss = self.forward_cycle_loss(self.generator_A2B, self.generator_B2A, realA)
        backward_loss = self.forward_cycle_loss(self.generator_B2A, self.generator_A2B, realB)

        # Track losses
        self.losses["cycle_loss_A2B"] = forward_loss
        self.losses["cycle_loss_B2A"] = backward_loss

        return forward_loss, backward_loss

    def forward(self, image: torch.Tensor, is_inverse: bool = True):
        if not is_inverse:
            return self.generator_A2B(image)
        return self.generator_B2A(image)

    def step_lr_schedulers(self):
        self.gen_lr_scheduler.step()
        self.discA_lr_scheduler.step()
        self.discB_lr_scheduler.step()

    # def train_model(self, real_imageA: torch.Tensor, real_imageB: torch.Tensor) -> dict:
    #     losses = dict()

    #     with torch.autograd.set_detect_anomaly(True):
    #         self.gen_optim.zero_grad()

    #         batch_size = real_imageA.shape[0]
    #         real_label = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.float32)
    #         fake_label = torch.full((batch_size, 1), 0, device=self.device, dtype=torch.float32)

    #         #########################################################
    #         # Update the generators with CycleGAN and Identity      #
    #         #########################################################
    #         # self.discriminator_A.set_requires_grad(False)
    #         # self.discriminator_B.set_requires_grad(False)

    #         # Identity loss
    #         losses["identity_A2B"], losses["identity_B2A"] = self.identity_loss(real_imageA, real_imageB)

    #         # Adversarial Loss
    #         losses["genB2A_adv"] = self.adversarial_loss_func(
    #             self.discriminator_A(self.generator_B2A(real_imageB)), real_label)

    #         losses["genA2B_adv"] = self.adversarial_loss_func(
    #             self.discriminator_B(self.generator_A2B(real_imageA)), real_label)

    #         # Cycle GAN loss
    #         losses["cycle_A2B"], losses["cycle_B2A"] = self.cycle_loss(real_imageA, real_imageB)

    #         total_gen_loss = losses["cycle_A2B"] + losses["cycle_B2A"] + \
    #             losses["identity_A2B"] + losses["identity_B2A"] +\
    #             losses["genB2A_adv"] + losses["genA2B_adv"]
    #         total_gen_loss.backward()
    #         self.gen_optim.step()

    #         #####################################################
    #         # Update the discriminators using adversarial loss  #
    #         #####################################################
    #         self.discA_optim.zero_grad()
    #         self.discB_optim.zero_grad()

    #         # self.discriminator_A.set_requires_grad(True)
    #         # self.discriminator_B.set_requires_grad(True)

    #         losses["discA_adversarial"] = self.adversarial_loss(self.discriminator_A, self.discA_optim, real_imageA,
    #                                                             self.generator_B2A.last_generated.pop().detach())
    #         losses["discA_adversarial"].backward()
    #         losses["discB_adversarial"] = self.adversarial_loss(self.discriminator_B, self.discB_optim, real_imageB,
    #                                                             self.generator_A2B.last_generated.pop().detach())
    #         losses["discB_adversarial"].backward()
    #         self.discA_optim.step()
    #         self.discB_optim.step()
    #         self.gen_optim.step()

    #         self.gen_lr_scheduler.step()
    #         self.discA_lr_scheduler.step()
    #         self.discB_lr_scheduler.step()

    #     return losses

    def update_generators(self, real_image_A: torch.Tensor, real_image_B: torch.Tensor,
                          real_label: torch.Tensor):
        self.gen_optim.zero_grad()

        # Identity Loss
        idt_loss_A2B, idt_loss_B2A = self.identity_loss(real_image_A, real_image_B)

        # Adversarial Loss
        adv_loss_A2B, adv_loss_B2A = self.adversarial_loss_generator(real_image_A, real_image_B, real_label)

        # Cycle Loss
        forward_loss, backward_loss = self.cycle_loss(real_image_A, real_image_B)

        # Calculate total generators loss
        total_gen_loss = idt_loss_A2B + idt_loss_B2A + adv_loss_A2B + adv_loss_B2A + forward_loss + backward_loss

        total_gen_loss.backward()
        self.gen_optim.step()

    def update_discriminator(self, discriminator: Discriminator, optimizer, real_image: torch.Tensor,
                             fake_image: torch.Tensor, real_label: torch.Tensor, fake_label: torch.Tensor):
        optimizer.zero_grad()

        real_image_loss = self.adversarial_loss_func(discriminator(real_image), real_label)
        fake_image_loss = self.adversarial_loss_func(discriminator(fake_image), fake_label)
        total_loss = (real_image_loss + fake_image_loss) / 2

        total_loss.backward()
        optimizer.step()

        self.losses[f"adv_loss_{discriminator.__class__.__name__}"] = total_loss
        return total_loss

    def train_model(self, real_imageA: torch.Tensor, real_imageB: torch.Tensor) -> dict:
        batch_size = real_imageA.shape[0]
        real_label = torch.full(
            (batch_size, 1), 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full(
            (batch_size, 1), 0, device=self.device, dtype=torch.float32)

        with torch.autograd.set_detect_anomaly(True):
            self.gen_optim.zero_grad()

            self.update_generators(real_imageA, real_imageB,
                                   real_label)
            self.update_discriminator(self.discriminator_A, self.discA_optim, real_imageA,
                                      self.generator_B2A.get_fake_image(), real_label, fake_label)
            self.update_discriminator(self.discriminator_B, self.discB_optim, real_imageB,
                                      self.generator_A2B.get_fake_image(), real_label, fake_label)

        return self.losses
