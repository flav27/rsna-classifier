import pytorch_lightning
import pandas as pd
import torch
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torchvision.transforms import transforms

from datasets.single_image_dataset import SingleImageDataset
from datasets.single_image_dataset_oversample import SingleImageDatasetOversample
from modelling.predict_model import PredictModel
from modelling.rsna_classifier import RsnaClassifier
from preprocessing.data_transform import DataTransform
from utils.utils import generate_splits, upsample_single_class, test_data_leakage

SEED = 1322
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def generate_data(train_csv_path, input_image_path, out_images_path, data_preprocess=True):
    dt = DataTransform(data=pd.read_csv(train_csv_path), in_path=input_image_path, out_path=out_images_path)
    if data_preprocess:
        dt.pre_process_and_save()
    return dt.data


def train_and_evaluate_models(data, train_folds, out_model_path):
    for i in range(0, len(train_folds)):
        train_fold_id = i
        print(f"training train fold id {train_fold_id}")
        test_data_leakage(data, train_folds)

        train_data = data.iloc[train_folds[train_fold_id][0]]
        val_data = data.iloc[train_folds[train_fold_id][1]]
        train_data = upsample_single_class(train_data, xup=5)

        cancer_p_nr = len(train_data.loc[train_data['cancer'] == 1])
        cancer_n_nr = len(train_data.loc[train_data['cancer'] == 0])
        cancer_n_sample_p = cancer_p_nr / cancer_n_nr

        train_transforms = transforms.Compose([
            transforms.Resize((1536, 768), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.9, 1), shear=None,
                                    interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0, hue=0),
            transforms.RandomAdjustSharpness(3, p=0.5),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ToTensor()
        ])

        val_image_transforms = transforms.Compose([
            transforms.Resize((1536, 768), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])

        train_dataset = SingleImageDatasetOversample(data=train_data, image_transform=train_transforms,
                                                     negative_percentage=cancer_n_sample_p)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                      pin_memory=False, num_workers=2, drop_last=True)

        val_dataset = SingleImageDataset(data=val_data, image_transform=val_image_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

        lr_logger = LearningRateMonitor()

        model_path = f"{out_model_path}/{train_fold_id}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_path,
            filename="best_model",
            verbose=True,
            monitor='val_pfbeta',
            mode='max'
        )

        tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
            f"{out_model_path}/{train_fold_id}")
        trainer = Trainer(
            logger=tb_logger,
            max_epochs=25,
            gpus=1,
            callbacks=[lr_logger, checkpoint_callback],
            accumulate_grad_batches=2,
            precision=16
        )

        model = RsnaClassifier()
        model.cuda()

        print(model)
        print(f"Number of parameters in network: {model.size() / 1e3:.1f}k")

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        model = PredictModel(model_path=model_path, model_cls=RsnaClassifier)
        model.inference(val_data)
        model.calculate_thresholds()


def generate_data(train_csv_path, input_image_path, out_images_path, data_preprocess=True):
    dt = DataTransform(data=pd.read_csv(train_csv_path), in_path=input_image_path, out_path=out_images_path)
    if data_preprocess:
        dt.pre_process_and_save()
    return dt.data


if __name__ == '__main__':

    TRAIN_CSV_PATH = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
    IN_IMAGE_PATH = "/kaggle/input/rsna-breast-cancer-detection/train_images"
    OUT_IMAGES_PATH = "/kaggle/working/rsna_png_1536/transformed_data"
    OUT_MODEL_PATH = "/kaggle/working/rsna_classifier/"
    CV_SPLITS = 5

    data = generate_data(TRAIN_CSV_PATH, IN_IMAGE_PATH, OUT_IMAGES_PATH)
    train_folds = generate_splits(data, CV_SPLITS)
    train_and_evaluate_models(data, train_folds, OUT_MODEL_PATH)


