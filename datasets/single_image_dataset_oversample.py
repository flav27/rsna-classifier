import PIL
import numpy as np

from torch.utils.data import Dataset, RandomSampler


class SingleImageDatasetOversample(Dataset):

    MAX_UPSAMPLE = 5

    def __init__(self, data, image_transform=None, negative_percentage=1, positive_percentage=1):
        """
        Extension of SingleImageDataset that supports oversampling/undersampling by percentages
        For each epoch a new subset from the negative class is resampled
        """

        self.data = data
        self.data.reset_index(drop=True, inplace=True)
        self.image_transform = image_transform
        self.labels = self._encode_target()
        assert positive_percentage < SingleImageDatasetOversample.MAX_UPSAMPLE
        self.negative_percentage = negative_percentage
        self.positive_percentage = positive_percentage
        self.sampler = None
        self.final_subset = None
        self.neg_indices, self.pos_indices = self.get_indices()
        self.update_sampler()

    def __len__(self):
        return len(self.final_subset)

    def __getitem__(self, idx):
        data_info = self.data.iloc[self.final_subset[idx]]
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

    def _encode_target(self):
        labels = self.data['cancer'].values.astype(np.int64).reshape(-1)
        return labels

    def get_indices(self):
        neg_indices = np.where(self.labels == 0)[0]
        pos_indices = np.where(self.labels == 1)[0]
        return neg_indices, pos_indices

    def update_sampler(self):
        np.random.shuffle(self.neg_indices)
        subset_size = int(self.negative_percentage * len(self.neg_indices)) - 1
        negative_subset = self.neg_indices[:subset_size]
        np.random.shuffle(self.pos_indices)
        positive_subset = self.pos_indices
        if self.positive_percentage > 1:
            extra_indices_size = int(len(positive_subset) * (self.positive_percentage - 1))
            positive_subset = np.concatenate((positive_subset[:extra_indices_size], self.pos_indices))
        self.final_subset = np.concatenate((positive_subset, negative_subset))
        np.random.shuffle(self.final_subset)
        self.sampler = RandomSampler(self.final_subset)




