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
        The identity loss is computed by comparing the output of the discriminator against real and fake images with
        known labels. Only affects the discriminator.
        :return: Loss value
        """
        # self.switch_mode()

        identity_A2B = self.identity_loss_func(
            self.generator_B2A(real_image_A), real_image_A)
        identity_B2A = self.identity_loss_func(
            self.generator_A2B(real_image_B), real_image_B)

        # loss = (identity_A2B + identity_B2A) * self.lambda_param / 2

        # loss.backward()
        return identity_A2B, identity_B2A

    def adversarial_loss(self, discriminator: Discriminator, optimizer,
                         real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        The adversarial loss affects the generator and the discriminator. Its goal is to minimize the loss of the
        discriminator given an input from the generator, intuitively, to trick the generator.
        :return: Loss value
        """
        # self.switch_mode()

        # disc_prediction_real = discriminator(real)
        # disc_loss_real = self.adversarial_loss_func(disc_prediction_real,
        #                                             torch.full(disc_prediction_real.shape, 1, device=self.device).to(
        #                                                 torch.float32))
        #
        # disc_prediction_fake = discriminator(fake)
        # disc_loss_fake = self.adversarial_loss_func(disc_prediction_fake,
        #                                             torch.full(disc_prediction_fake.shape, 1, device=self.device).to(
        #                                                 torch.float32))
        # disc_loss = disc_loss_real + disc_loss_fake
        # disc_loss.backward()
        batch_size = real.shape[0]
        real_label = torch.full(
            (batch_size, 1), 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full(
            (batch_size, 1), 0, device=self.device, dtype=torch.float32)

        adversarial = self.adversarial_loss_func(discriminator(real), real_label) + \
            self.adversarial_loss_func(discriminator(fake), fake_label)
        # adversarial.backward()
        # optimizer.step()
        return adversarial

    def forward_cycle_loss(self, generator: Generator, inv_generator: Generator, optimizer,
                           real: torch.Tensor) -> torch.Tensor:
        self.switch_mode()

        fake_image = generator(real).to(self.device).detach()
        converted_back = inv_generator(fake_image).to(self.device)

        loss = self.cycle_loss_func(converted_back, real) * self.lambda_param
        # loss.backward()

        optimizer.step()
        return loss

    def cycle_loss(self, realA: torch.Tensor, realB: torch.Tensor) -> tuple:
        # self.switch_mode()
        forward_loss = self.forward_cycle_loss(self.generator_A2B, self.generator_B2A, self.gen_optim,
                                               realA) * self.lambda_param
        backward_loss = self.forward_cycle_loss(self.generator_B2A, self.generator_A2B, self.gen_optim,
                                                realB) * self.lambda_param
        return forward_loss, backward_loss

    def forward(self, image: torch.Tensor, is_inverse: bool = True):
        # self.switch_mode()

        if not is_inverse:
            return self.generator_A2B(image)
        return self.generator_B2A(image)

    def switch_mode(self):
        # if not self.training:
        #     self.generator_A2B.eval()
        #     self.generator_B2A.eval()
        #     self.discriminator_B.eval()
        #     self.discriminator_A.eval()
        # else:
        #     self.generator_A2B.train()
        #     self.generator_B2A.train()
        #     self.discriminator_B.train()
        #     self.discriminator_A.train()
        pass

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
                          real_label: torch.Tensor, losses: dict):

        self.gen_optim.zero_grad()

        # Identity Loss
        identity_image_A = self.generator_B2A(real_image_A)
        loss_identity_A = self.identity_loss_func(
            identity_image_A, real_image_A) * self.lambda_param / 2

        identity_image_B = self.generator_A2B(real_image_B)
        loss_identity_B = self.identity_loss_func(
            identity_image_B, real_image_B) * 5.0

        # Adversarial Loss
        # GAN loss D_A(G_A(A))
        fake_image_A = self.generator_B2A(real_image_B)
        fake_output_A = self.discriminator_A(fake_image_A)
        real_label = torch.full(fake_output_A.shape, 1, device=self.device, dtype=torch.float32)
        loss_GAN_B2A = self.adversarial_loss_func(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = self.generator_A2B(real_image_A)
        fake_output_B = self.discriminator_B(fake_image_B)
        loss_GAN_A2B = self.adversarial_loss_func(fake_output_B, real_label)

        # Cycle Loss
        recovered_image_A = self.generator_B2A(fake_image_B)
        loss_cycle_ABA = self.cycle_loss_func(
            recovered_image_A, real_image_A) * self.lambda_param

        recovered_image_B = self.generator_A2B(fake_image_A)
        loss_cycle_BAB = self.cycle_loss_func(
            recovered_image_B, real_image_B) * self.lambda_param

        # Store all losses
        losses["gen_A2B_idt"] = loss_identity_A
        losses["gen_B2A_idt"] = loss_identity_B
        losses["gen_A2B_adv"] = loss_GAN_A2B
        losses["gen_B2A_adv"] = loss_GAN_B2A
        losses["loss_cycle_ABA"] = loss_cycle_ABA
        losses["loss_cycle_BAB"] = loss_cycle_BAB

        total_gen_loss = loss_identity_A + loss_identity_B + loss_GAN_A2B + \
            loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        total_gen_loss.backward()
        self.gen_optim.step()

    def update_discriminator_A(self, real_image_A: torch.Tensor, real_label, fake_label, losses):
        # Set D_A gradients to zero
        self.discA_optim.zero_grad()

        # Real A image loss
        real_output_A = self.discriminator_A(real_image_A)

        real_label = torch.full(real_output_A.shape, 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full(real_output_A.shape, 0, device=self.device, dtype=torch.float32)

        errD_real_A = self.adversarial_loss_func(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = self.generator_B2A.last_generated.pop()
        fake_output_A = self.discriminator_A(fake_image_A.detach())
        errD_fake_A = self.adversarial_loss_func(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        self.discA_optim.step()

        losses["discA_adv"] = errD_A

    def update_discriminator_B(self, real_image_B: torch.Tensor, real_label, fake_label, losses):
        # Set D_B gradients to zero
        self.discB_optim.zero_grad()

        # Real B image loss
        real_output_B = self.discriminator_A(real_image_B)

        real_label = torch.full(real_output_B.shape, 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full(real_output_B.shape, 0, device=self.device, dtype=torch.float32)

        errD_real_B = self.adversarial_loss_func(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = self.generator_B2A.last_generated.pop()
        fake_output_B = self.discriminator_A(fake_image_B.detach())
        errD_fake_B = self.adversarial_loss_func(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        self.discB_optim.step()

        losses["discB_adv"] = errD_A

    def train_model(self, real_imageA: torch.Tensor, real_imageB: torch.Tensor) -> dict:
        batch_size = real_imageA.shape[0]
        losses = dict()

        with torch.autograd.set_detect_anomaly(True):
            self.gen_optim.zero_grad()

            real_label = torch.full(
                (batch_size, 1), 1, device=self.device, dtype=torch.float32)
            fake_label = torch.full(
                (batch_size, 1), 0, device=self.device, dtype=torch.float32)

            self.update_generators(real_imageA, real_imageB,
                                   real_label, losses)
            self.update_discriminator_A(real_imageA, real_label, fake_label, losses)
            self.update_discriminator_B(real_imageB, real_label, fake_label, losses)

        return losses
