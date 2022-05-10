import torch
from torch.cuda.amp import GradScaler
from torchvision import transforms
from data.dataset import Pix2PixDataset
from pix2pix.pix2pix import Pix2Pix

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
        generator_opt=torch.optim.Adam(model.generator.parameters(), 2e-4, betas=(0.5, 0.999)),
        discriminator_opt=torch.optim.Adam(model.discriminator.parameters(), 2e-4, betas=(0.5, 0.999)),
        g_scaler=GradScaler(),
        d_scaler=GradScaler()
    )

    model.fit(train_dataset, val_dataset)
