import os

import torch.nn as nn
import torch
from os import path

from cyclegan.utils import *

from .generator import Generator
from .discriminator import Discriminator
from .base_model import BaseModel


class CycleGAN(BaseModel):
    def __init__(self, lr: float, lambda_param: float, continue_learning: bool, counters: dict, device):
        super().__init__()

        self.lambda_param = lambda_param
        self.device = device

        # Define the generators and discriminators
        self.generator_A2B = Generator().to(device)
        self.generator_B2A = Generator().to(device)
        self.discriminator_A = Discriminator().to(device)
        self.discriminator_B = Discriminator().to(device)

        self.init_models(continue_learning, counters)

        # Define the loss functions
        self.identity_loss_func = torch.nn.L1Loss()
        self.adversarial_loss_func = torch.nn.MSELoss()
        self.cycle_loss_func = torch.nn.MSELoss()

        ##############
        # Optimizers #
        ##############
        self.genA2B_optim = torch.optim.Adam(self.generator_A2B.parameters(), lr=lr)
        self.genB2A_optim = torch.optim.Adam(self.generator_B2A.parameters(), lr=lr)
        self.discA_optim = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr)
        self.discB_optim = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr)

        # Defining Decay LR - Gradually changing the LR
        self.genA2B_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.genA2B_optim, gamma=0.5)
        self.genB2A_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.genB2A_optim, gamma=0.5)
        self.discA_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discA_optim, gamma=0.5)
        self.discB_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discB_optim, gamma=0.5)

    def init_models(self, cont: bool, counters: dict) -> None:
        # Load the latest models if we want to continue learning
        if cont and len(os.listdir(GEN_A2B_PATH)) > 0:
            self.generator_A2B.load_state_dict(torch.load(path.join(GEN_A2B_PATH, str(counters["genA2B"]) + ".pth")))
            self.generator_B2A.load_state_dict(torch.load(path.join(GEN_B2A_PATH, str(counters["genB2A"]) + ".pth")))
            self.discriminator_A.load_state_dict(torch.load(path.join(DISC_A_PATH, str(counters["discA"]) + ".pth")))
            self.discriminator_B.load_state_dict(torch.load(path.join(DISC_B_PATH, str(counters["discB"]) + ".pth")))
        else:
            self.generator_A2B.apply(weights_init)
            self.generator_B2A.apply(weights_init)
            self.discriminator_A.apply(weights_init)
            self.discriminator_B.apply(weights_init)

    def identity_loss(self, discriminator: Discriminator, optimizer, real_image: torch.Tensor,
                      fake_image: torch.Tensor) -> torch.Tensor:
        """
        The identity loss is computed by comparing the output of the discriminator against real and fake images with
        known labels. Only affects the discriminator.
        :return: Loss value
        """
        # Real
        real_pred = discriminator(real_image).to(self.device)
        loss_real = self.identity_loss_func(real_pred, torch.full(real_pred.shape, 1).to(torch.float32))
        # Fake
        fake_image = fake_image.to(self.device)
        fake_pred = discriminator(fake_image)
        loss_fake = self.identity_loss_func(fake_pred, torch.full(real_pred.shape, 0).to(torch.float32))

        # Compute the loss
        loss = (loss_fake + loss_real)
        loss.backward()
        optimizer.step()
        return loss

    def adversarial_loss(self, generator: Generator, discriminator: Discriminator, optimizer,
                         real: torch.Tensor) -> torch.Tensor:
        """
        The adversarial loss affects the generator. Its goal is to minimize the loss of the discriminator given an
        input from the generator, intuitively, to trick the generator.
        :return: Loss value
        """
        fake = generator(real).to(self.device)
        disc_prediction = discriminator(fake)
        disc_loss = self.adversarial_loss_func(disc_prediction,
                                               torch.full(disc_prediction.shape, 1, device=self.device).to(
                                                   torch.float32))
        disc_loss.backward()

        optimizer.step()
        return disc_loss

    def forward_cycle_loss(self, generator: Generator, inv_generator: Generator, optimizer,
                           real: torch.Tensor) -> torch.Tensor:
        fake_image = generator(real).to(self.device)
        converted_back = inv_generator(fake_image).to(self.device)
        loss = self.cycle_loss_func(converted_back, real)
        loss.backward()

        optimizer.step()
        return loss

    def cycle_loss(self, realA: torch.Tensor, realB: torch.Tensor) -> torch.Tensor:
        forward_loss = self.forward_cycle_loss(self.generator_A2B, self.generator_B2A, self.genA2B_optim,
                                               realA) * self.lambda_param
        backward_loss = self.forward_cycle_loss(self.generator_B2A, self.generator_A2B, self.genB2A_optim,
                                                realB) * self.lambda_param
        return forward_loss + backward_loss

    def forward(self, image: torch.Tensor, is_inverse: bool = True):
        if not is_inverse:
            return self.generator_A2B(image)
        return self.generator_B2A(image)

    def step_lr_schedulers(self):
        self.genA2B_lr_scheduler.step()
        self.genB2A_lr_scheduler_lr_scheduler.step()
        self.discA_lr_scheduler_lr_scheduler.step()
        self.discB_lr_scheduler.step()

    def train_model(self, real_imageA: torch.Tensor, real_imageB: torch.Tensor) -> dict:
        losses = dict()

        self.genA2B_optim.zero_grad()
        self.genB2A_optim.zero_grad()
        self.discA_optim.zero_grad()
        self.discB_optim.zero_grad()

        #########################################################
        # Update the generators with CycleGAN and Adversarial   #
        #########################################################
        # self.discriminator_A.set_requires_grad(False)
        # self.discriminator_B.set_requires_grad(False)

        # Adversarial loss
        losses["genA2B_adversarial"] = self.adversarial_loss(self.generator_A2B, self.discriminator_B,
                                                             self.genA2B_optim, real_imageA)
        losses["genB2A_adversarial"] = self.adversarial_loss(self.generator_B2A, self.discriminator_A,
                                                             self.genB2A_optim, real_imageB)
        # Cycle GAN loss
        losses["cycle_loss"] = self.cycle_loss(real_imageA, real_imageB)

        #####################################################
        # Update the discriminators using Identity loss     #
        #####################################################
        # self.discriminator_A.set_requires_grad(True)
        # self.discriminator_B.set_requires_grad(True)

        fake_imageA = self.generator_B2A.last_generated.pop().detach()
        losses["discA_identity"] = self.identity_loss(self.discriminator_A, self.discA_optim, real_imageA,
                                                      fake_imageA)

        fake_imageB = self.generator_A2B.last_generated.pop().detach()
        losses["discB_identity"] = self.identity_loss(self.discriminator_B, self.discB_optim, real_imageB,
                                                      fake_imageB)

        return losses
