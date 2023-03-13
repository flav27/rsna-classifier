import PIL
import numpy as np

from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    def __init__(self, data, image_transform=None):
        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_info = self.data.iloc[idx]
        train_data = {'image': None, 'index': None}
        image = self.read_image(data_info['out_file_paths'])
        if self.image_transform:
            image = self.image_transform(image)
        train_data['image'] = image
        train_data['index'] = idx
        label = np.array([data_info['cancer']]).astype(dtype=np.float32)
        return train_data, label

    @staticmethod
    def read_image(image_path):
        img = PIL.Image.open(image_path)
        gray_img = img.convert("L")
        return gray_img



