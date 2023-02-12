import random
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from datasets.single_image_dataset import SingleImageDataset
from grouped_stratified_k_fold_binary import GroupStratifiedShuffleSplitBinary


def generate_splits(df, n_splits=4):
    """
    Generates Cross Validation splits using GroupStratifiedShuffleSplitBinary, an extension to StratifiedKFold
    that also uses in consideration the groups. Needed here because there are multiple images per patient, and I
    wanted to split in train/val using the patient_id, so all images for a specific patient go either in train or
    val datasets
    """

    y = df['cancer']
    df['age_bins'] = pd.cut(df['age'], bins=3)
    X = df.drop(['cancer', 'patient_id', 'image_id', 'age', 'in_file_paths', 'out_file_paths'], axis=1)
    splitter = GroupStratifiedShuffleSplitBinary(y, df['patient_id'], n_splits=n_splits, frac_train=0.9)
    folds = []
    for train_index, test_index in splitter.split(X, y, df['patient_id']):
        folds.append((train_index, test_index))
    return folds


def test_data_leakage(df, folds):
    """
    Tests dataset leakage, no image should be both in train and val
    """

    for i, (train_index, test_index) in enumerate(folds):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        for col in df.columns:
            if col == 'out_file_paths':
                train_col_vals = set(train_data[col])
                test_col_vals = set(test_data[col])
                overlap = train_col_vals.intersection(test_col_vals)
                if overlap:
                    print("WARNING: Fold {} has data leakage in column {}".format(i, col))
                else:
                    print("Fold {} has no data leakage in column {}".format(i, col))


def get_mean_std(train_dataloader):
    mean = torch.zeros(1).to(torch.device("cuda"))
    std = torch.zeros(1).to(torch.device("cuda"))

    for data, i in tqdm(train_dataloader):
        images = data['image'].to(torch.device("cuda"))
        mean += images.mean(dim=(0, 2, 3))
        std += images.std(dim=(0, 2, 3))

    mean /= len(train_dataloader)
    std /= len(train_dataloader)
    return mean, std


def find_mean_and_std(data):

    val_image_transforms = transforms.Compose([
        transforms.Resize((1536, 768), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
    ])
    num_workers = 2
    if os.name == 'nt':
        num_workers = 0

    train_dataset = SingleImageDataset(data=data, image_transform=val_image_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=num_workers, drop_last=False)
    return get_mean_std(train_dataloader)


def visualize(dataset, dual_image=False):
    # Get the indices of random images from the dataset
    indices = random.sample(range(len(dataset)), 20)

    if dual_image:
        # Get the images corresponding to the indices
        cc_images = [dataset[i][0]['cc'] for i in indices]
        mlo_images = [dataset[i][0]['mlo'] for i in indices]

        # create a grid of images
        img_grid = vutils.make_grid(cc_images + mlo_images, nrow=10)
    else:
        # Get the images corresponding to the indices
        images = [dataset[i][0]['image'] for i in indices]
        img_grid = vutils.make_grid(images, nrow=10)

    # show the grid of images
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(img_grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def load_inference_model(model_cls, model_path):
    checkpoint = torch.load(model_path)
    model = model_cls()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def upsample_single_class(dataset, xup):
    """
    Up-sample the positive class as this is needed in a highly imbalanced dataset with only around 2% of data being
    in the positive class
    """
    dataset_neg = dataset.loc[dataset['cancer'] == 0]
    dataset_minority = dataset.loc[dataset['cancer'] == 1]
    upsample_df = dataset_neg
    for i in range(0, xup):
        upsample_df = pd.concat([upsample_df, dataset_minority], axis=0, ignore_index=True)
        upsample_df.reset_index(drop=True, inplace=True)
    return upsample_df




