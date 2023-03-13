import os

import pandas as pd

from preprocessing.data_transform import DataTransform


def generate_data(train_csv_path, input_image_path, out_images_path):
    dt = DataTransform(data=pd.read_csv(train_csv_path), in_path=input_image_path, out_path=out_images_path)
    dt.data = dt.data[dt.data['in_file_paths'].apply(os.path.exists)]
    dt.pre_process_and_save()


if __name__ == '__main__':
    TRAIN_CSV_PATH = "E:/kaggle/RSNAScreeningMammography/data/train.csv"
    IN_IMAGE_PATH = "E:/kaggle/RSNAScreeningMammography/data/train_images"
    OUT_IMAGES_PATH = "E:/kaggle/RSNAScreeningMammography/data/selection_transformed_data_processed"

    generate_data(TRAIN_CSV_PATH, IN_IMAGE_PATH, OUT_IMAGES_PATH)