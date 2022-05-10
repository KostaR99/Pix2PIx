import os
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pix2pix.constants import BATCH_SIZE, DEVICE, L1_LAMBDA, NUM_EPOCHS, NUM_WORKERS
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator

# Wrapper clas for our gan
class Pix2Pix:
    def __init__(self) -> None:
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.discriminator_opt = None
        self.generator_opt = None
        self.d_scaler = None
        self.g_scaler = None

    def compile(self, discriminator_opt, generator_opt):

        self.discriminator_opt = discriminator_opt
        self.generator_opt = generator_opt

    def __save_checkpoint(self, model, checkpoint_name: str):
        print("Saving the checkpoint...")
        os.makedirs("./checkpoints", exist_ok=True)
        checkpoint_path = os.path.join("./checkpoints", checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved! Checkpoint path is: {checkpoint_path}")

    def __save_some_exapmles(self, data_loader, current_epoch):
        print("Saving examples...")
        self.generator.eval()

        os.makedirs("./results", exist_ok=True)

        x, y = next(iter(data_loader))

        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.no_grad():
            results = self.generator(x)
            results = results.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            for idx in range(len(results)):
                result = np.moveaxis(results[idx], 0, -1)
                y = np.moveaxis(y[idx], 0, -1)

                res = np.concatenate((result, y), axis=1)
                image = Image.fromarray((res * 255).astype(np.uint8))
                image.save(f"./results/epoch_{current_epoch}_{idx}.png")
        self.generator.train()
        print("Saved!")

    def __train_step(self, data):
        discriminator_losses = []
        generator_losses = []
        loop = tqdm(data)
        for _, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # train discriminator
            y_fake = self.generator(x)
            D_real = self.discriminator(x, y)
            D_fake = self.discriminator(x, y_fake.detach())

            D_real_loss = self.bce(D_real, torch.ones_like(D_real))
            D_fake_loss = self.bce(D_fake, torch.zeros_like(D_fake))

            D_loss = (D_real_loss + D_fake_loss) / 2

            discriminator_losses.append(D_loss)
            self.discriminator_opt.zero_grad()
            D_loss.backward()
            self.discriminator_opt.step()

            # train generator
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

    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ):

        if not num_epochs or num_epochs <= 0:
            num_epochs = NUM_EPOCHS

        if not batch_size or batch_size <= 0:
            batch_size = BATCH_SIZE

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        if validation_dataset:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=NUM_WORKERS
            )

        dummy_tensor = torch.randn((1, 3, 256, 256))
        self.generator(dummy_tensor)
        self.discriminator(dummy_tensor, dummy_tensor)

        self.generator.normal_weight_init()
        self.discriminator.normal_weight_init()

        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)

        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch}")
            self.__train_step(data=train_loader)

            if epoch % 10 == 0:
                self.__save_checkpoint(
                    model=self.generator,
                    checkpoint_name=f"generator_checkpoint_epoch_{epoch}")
                self.__save_checkpoint(
                    model=self.discriminator,
                    checkpoint_name=f"discriminator_checkpoint_epoch_{epoch}")

                self.__save_some_exapmles(validation_loader, epoch)
