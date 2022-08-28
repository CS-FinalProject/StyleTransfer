import os

import torch.nn as nn
import torch
from os import path
from itertools import chain

from cyclegan.utils import *

from .generator import Generator
from .discriminator import Discriminator
from .base_model import BaseModel
# from .losses import AdversarialLossFunction
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
                                          lr=lr)
        self.discA_optim = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr)
        self.discB_optim = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr)

        # Defining Decay LR - Gradually changing the LR
        self.gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optim, step_size=decay_epoch, gamma=0.1)
        self.discA_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.discA_optim, step_size=decay_epoch, gamma=0.1)
        self.discB_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.discB_optim, step_size=decay_epoch, gamma=0.1)

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
        self.switch_mode()

        identity_B2A = self.identity_loss_func(self.generator_A2B(real_image_B), real_image_B)
        identity_A2B = self.identity_loss_func(self.generator_B2A(real_image_A), real_image_A)

        loss = (identity_A2B + identity_B2A) * self.lambda_param / 2

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
        real_label = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=self.device, dtype=torch.float32)

        adversarial = self.adversarial_loss_func(discriminator(real), real_label) + \
                self.adversarial_loss_func(discriminator(fake), fake_label)
        # adversarial.backward()
        # optimizer.step()
        return adversarial

    def forward_cycle_loss(self, generator: Generator, inv_generator: Generator, optimizer,
                           real: torch.Tensor) -> torch.Tensor:
        self.switch_mode()

        fake_image = generator(real).to(self.device)
        converted_back = inv_generator(fake_image).to(self.device)

        loss = self.cycle_loss_func(converted_back, real) * self.lambda_param
        # loss.backward()

        optimizer.step()
        return loss

    def cycle_loss(self, realA: torch.Tensor, realB: torch.Tensor) -> tuple:
        self.switch_mode()
        forward_loss = self.forward_cycle_loss(self.generator_A2B, self.generator_B2A, self.gen_optim,
                                               realA) * self.lambda_param
        backward_loss = self.forward_cycle_loss(self.generator_B2A, self.generator_A2B, self.gen_optim,
                                                realB) * self.lambda_param
        return forward_loss, backward_loss

    def forward(self, image: torch.Tensor, is_inverse: bool = True):
        self.switch_mode()

        if not is_inverse:
            return self.generator_A2B(image)
        return self.generator_B2A(image)

    def switch_mode(self):
        if not self.training:
            self.generator_A2B.eval()
            self.generator_B2A.eval()
            self.discriminator_B.eval()
            self.discriminator_A.eval()
        else:
            self.generator_A2B.train()
            self.generator_B2A.train()
            self.discriminator_B.train()
            self.discriminator_A.train()

    def step_lr_schedulers(self):
        self.gen_lr_scheduler.step()
        self.discA_lr_scheduler.step()
        self.discB_lr_scheduler.step()

    def train_model(self, real_imageA: torch.Tensor, real_imageB: torch.Tensor) -> dict:
        losses = dict()

        self.gen_optim.zero_grad()

        #########################################################
        # Update the generators with CycleGAN and Identity      #
        #########################################################
        # self.discriminator_A.set_requires_grad(False)
        # self.discriminator_B.set_requires_grad(False)

        # Identity loss
        losses["identity_A2B"], losses["identity_B2A"] = self.identity_loss(real_imageA, real_imageB)

        # Cycle GAN loss
        losses["cycle_A2B"], losses["cycle_B2A"] = self.cycle_loss(real_imageA, real_imageB)

        total_gen_loss = losses["cycle_A2B"] + losses["cycle_B2A"] + losses["identity_A2B"] + losses["identity_B2A"]
        total_gen_loss.backward()
        self.gen_optim.step()

        #####################################################
        # Update the discriminators using adversarial loss  #
        #####################################################
        self.discA_optim.zero_grad()
        self.discB_optim.zero_grad()

        # self.discriminator_A.set_requires_grad(True)
        # self.discriminator_B.set_requires_grad(True)

        losses["discA_adversarial"] = self.adversarial_loss(self.discriminator_A, self.discA_optim, real_imageA, 
                                                            self.generator_B2A.last_generated.pop().detach())
        losses["discA_adversarial"].backward()
        losses["discB_adversarial"] = self.adversarial_loss(self.discriminator_B, self.discB_optim, real_imageB,
                                                            self.generator_A2B.last_generated.pop().detach())
        losses["discB_adversarial"].backward()
        self.discA_optim.step()
        self.discB_optim.step()
        self.gen_optim.step()

        self.gen_lr_scheduler.step()
        self.discA_lr_scheduler.step()
        self.discB_lr_scheduler.step()

        return losses
