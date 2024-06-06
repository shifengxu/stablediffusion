import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.exists(root_dir):
            raise ValueError(f"Path not exist: {root_dir}")
        if not os.path.isdir(root_dir):
            raise ValueError(f"Path not dir: {root_dir}")
        name_arr = os.listdir(root_dir)
        name_arr.sort()
        self.img_path_arr = [os.path.join(root_dir, n) for n in name_arr]

    def __getitem__(self, index):
        img_path = self.img_path_arr[index]
        basename = os.path.basename(img_path)
        stem, ext = os.path.splitext(basename)
        img_id = int(stem)
        image = Image.open(img_path)
        img_rgb = image.convert("RGB")
        if self.transform:
            img_rgb = self.transform(img_rgb)
        img_np = np.array(img_rgb)
        return img_np, img_id

    def __len__(self):
        return len(self.img_path_arr)
