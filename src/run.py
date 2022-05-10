import torch
from torchvision import transforms
from data.dataset import Pix2PixDataset
from pix2pix.pix2pix import Pix2Pix
from src.pix2pix.constants import LEARNING_RATE

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = Pix2PixDataset(r"D:\\pix2pix2\\datasets\\maps\\train", transform)
    val_dataset = Pix2PixDataset(r"D:\\pix2pix2\\datasets\\maps\\val", transform)

    model = Pix2Pix()
    model.compile(
        generator_opt=torch.optim.Adam(model.generator.parameters(), LEARNING_RATE, betas=(0.5, 0.999)),
        discriminator_opt=torch.optim.Adam(model.discriminator.parameters(), LEARNING_RATE, betas=(0.5, 0.999)),
    )

    model.fit(train_dataset, val_dataset)
