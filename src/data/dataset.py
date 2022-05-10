import os
import random
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class Pix2PixDataset(Dataset):
    def __init__(self, root_dir: str, transforms) -> None:
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        img_file = self.file_list[index]
        img_path = os.path.join(self.root_dir, img_file)
        img = np.asarray(Image.open(img_path))
        img_width = img.shape[1] // 2

        input_img, output_img = Image.fromarray(img[:, :img_width, :]), Image.fromarray(img[:, img_width:, :])

        p = random.uniform(0, 1)
        if p < 0.5:
            input_img = ImageOps.mirror(input_img)
            output_img = ImageOps.mirror(output_img)

        input_img, output_img = self.transforms(input_img), self.transforms(output_img)

        return input_img, output_img


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Pix2PixDataset(r"D:\\pix2pix2\\datasets\\edges2shoes\\train", transform)
    img = dataset[0][1].cpu().detach().numpy()
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    plt.show()
