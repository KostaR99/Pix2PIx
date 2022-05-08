import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn as nn
from constants import DEVICE, L1_LAMBDA
from discriminator import Discriminator
from generator import Generator

# Wrapper clas for our gan
class Pix2Pix:
    def __init__(self) -> None:
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def compile(self, discriminator_opt, generator_opt, d_scaler, g_scaler):

        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt
        self.d_scaler = d_scaler
        self.g_scaler = g_scaler

    def __train_step(self, data):
        discriminator_losses = []
        generator_losses = []
        loop = tqdm(data)

        for _, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # train discriminator
            with autocast():
                y_fake = self.generator(x)
                D_real = self.discriminator(x, y)
                D_fake = self.discriminator(x, y_fake.detach())

                D_real_loss = self.bce(D_real, torch.ones_like(D_real))
                D_fake_loss = self.bce(D_fake, torch.ones_like(D_fake))

                D_loss = (D_real_loss + D_fake_loss) / 2

            discriminator_losses.append(D_loss)
            self.discriminator_opt.zero_grad()
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.discriminator_opt)
            self.d_scaler.update()

            # train generator
            with autocast():
                D_fake = self.discriminator(x, y_fake)
                G_loss = self.bce(D_fake, torch.ones_like(D_fake))
                l1_loss = self.l1(y_fake, y) * L1_LAMBDA
                G_loss += l1_loss

            generator_losses.append(G_loss)
            self.generator_opt.zero_grad()
            self.g_scaler.scale(G_loss).backward()
            self.g_scaler.step(self.generator_opt)
            self.g_scaler.update()

        mean_generator_loss = sum(generator_losses) / len(generator_losses)
        mean_discriminator_loss = sum(discriminator_losses) / len(discriminator_losses)

        return mean_generator_loss, mean_discriminator_loss
