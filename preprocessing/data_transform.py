import os
import cv2
import numpy as np
import dicomsdl
import pandas as pd

from multiprocessing.pool import Pool
from tqdm.auto import tqdm


class DataTransform:

    """
    Pre-Process DICOM files with the following methods:
        - photometric-interpretation:
          monochrome images can be represented as either "Monochrome 1" or "Monochrome 2"
        - windowing:
          also known as the "VOI-LUT" (Value of Interest-Look-Up Table) transform
        - normalize:
          dicom images have high intensities and here the image is normalized in range 0..255, to be later used
          by the DL models
        - crop:
          the region of interest is small by comparing it with the image, and cropping is applied to be able to keep
          the relevant information at a high quality
        - resize:
          dicom images have high resolution 5000x4000px for example, feeding CNN's at this resolution is not feasible
          resize the crop, and maintaining aspect ratio is essential
        - save:
          the images are saved in png format to maintain the original quality
    """

    HEIGHT = 1536
    WIDTH = 768

    def __init__(self, data, in_path, out_path):
        self.in_path = in_path
        self.out_path = out_path
        self.data = self.generate_paths(data)

    def generate_paths(self, data):
        data['in_file_paths'] = data.apply(lambda x: f'{self.in_path}/{x.patient_id}/{x.image_id}.dcm', axis=1)
        data['out_file_paths'] = data.apply(lambda x: f'{self.out_path}/{x.patient_id}_{x.image_id}.png',
                                            axis=1)
        return data

    def pre_process_and_save(self):
        os.makedirs(f'{self.out_path}', exist_ok=True)
        with Pool() as p:
            with tqdm(total=len(self.data)) as pbar:
                for _ in p.imap_unordered(self._pre_process_and_save_image, self.data.iterrows()):
                    pbar.update()

    def _crop(self, img):
        try:
            bin_img = self._binarize(img, threshold=5)
            contour = self._extract_contour(bin_img)
            img = self._erase_background(img, contour)
            x1, x2 = np.min(contour[:, :, 0]), np.max(contour[:, :, 0])
            y1, y2 = np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
            x1, x2 = int(0.99 * x1), int(1.01 * x2)
            y1, y2 = int(0.99 * y1), int(1.01 * y2)
        except:
            return img
        return img[y1:y2, x1:x2]

    @staticmethod
    def _binarize(img, threshold):
        return (img > threshold).astype(np.uint8)

    @staticmethod
    def _extract_contour(bin_img):
        contours, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        return contour

    @staticmethod
    def _erase_background(img, contour):
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        output = cv2.bitwise_and(img, mask)
        return output

    @staticmethod
    def _windowing(img, scan):
        function = scan.VOILUTFunction
        if type(scan.WindowWidth) == list:
            center = scan.WindowCenter[0]
            width = scan.WindowWidth[0]
        else:
            center = scan.WindowCenter
            width = scan.WindowWidth
        y_range = 2 ** scan.BitsStored - 1
        if function == 'SIGMOID':
            img = y_range / (1 + np.exp(-4 * (img - center) / width))
        else:  # LINEAR
            below = img <= (center - width / 2)
            above = img > (center + width / 2)
            between = np.logical_and(~below, ~above)
            img[below] = 0
            img[above] = y_range
            img[between] = ((img[between] - center) / width + 0.5) * y_range
        return img

    @staticmethod
    def _normalize_to_255(img):
        if img.max() != 0:
            img = img / img.max()
        img *= 255
        return img.astype(np.uint8)

    def _pre_process_and_save_image(self, index_row):
        index, row = index_row
        file_path = row['in_file_paths']
        scan = dicomsdl.open(file_path)
        img = scan.pixelData()
        try:
            img = self._fix_photometric_interpretation(img, scan)
            img = self._windowing(img, scan)
            img = self._normalize_to_255(img)
            img = self._crop(img)
            img = self._resize(img)
        except:
            img = np.zeros((self.HEIGHT, self.WIDTH), np.uint8)
            print("Encountered exception in process_and_save_image !!!")
        cv2.imwrite(row['out_file_paths'], img)

    def _resize(self, image):
        # Calculate the aspect ratio of the image
        try:
            aspect_ratio = image.shape[1] / image.shape[0]
            new_height = self.HEIGHT
            new_width = int(new_height * aspect_ratio)
            if new_width > self.WIDTH:
                new_width = self.WIDTH
                new_height = int(new_width / aspect_ratio)
            elif new_height > self.HEIGHT:
                new_height = self.HEIGHT
                new_width = int(new_height * aspect_ratio)
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            top = int((self.HEIGHT - new_height) / 2)
            bottom = self.HEIGHT - new_height - top
            left = int((self.WIDTH - new_width) / 2)
            right = self.WIDTH - new_width - left
            resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        except:
            print("Encountered exception in resize !!!")
            resized_image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_LINEAR)
        return resized_image

    @staticmethod
    def _fix_photometric_interpretation(image, ds):
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            image = image.max() - image
        elif ds.PhotometricInterpretation == 'MONOCHROME2':
            image = image - image.min()
        else:
            pass
        return image


if __name__ == '__main__':
    TRAIN_CSV_PATH = "E:/kaggle/RSNAScreeningMammography/data/train.csv"
    IN_IMAGE_PATH = "E:/kaggle/RSNAScreeningMammography/data/train_images"
    OUT_PATH = "E:/kaggle/RSNAScreeningMammography/data/transformed_data"

    data = pd.read_csv(TRAIN_CSV_PATH)
    dt = DataTransform(data, IN_IMAGE_PATH, OUT_PATH)
    dt.pre_process_and_save()