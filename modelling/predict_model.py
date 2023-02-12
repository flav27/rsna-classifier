import copy
import pandas as pd
import torch
import tqdm

from torchmetrics import Precision, Recall, Accuracy, Specificity
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets.single_image_dataset import SingleImageDataset
from utils.metrics import pfbeta_torch
from utils.utils import load_inference_model


class PredictModel:

    def __init__(self, model_path, model_cls, threshold=None):
        self.model = self.load_model(model_path, model_cls)
        self.dataset_name = "dataset.csv"
        self.raw_predictions = "raw_predictions.csv"
        self.threshold = threshold

    @staticmethod
    def load_model(model_path, model_cls):
        model = load_inference_model(model_cls=model_cls, model_path=model_path)
        model.eval()
        model.cuda()
        return model

    def inference(self, dataset):
        val_image_transforms = transforms.Compose([
            transforms.Resize((1536, 768), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
        ])
        val_dataset = SingleImageDataset(data=dataset, image_transform=val_image_transforms)
        data_loader = DataLoader(val_dataset, batch_size=64, num_workers=0, shuffle=False, drop_last=False)
        data_dict = {'index': [], 'predictions': []}
        progress_bar = tqdm.tqdm(desc="Predict", unit=" batches", total=len(data_loader))
        with torch.no_grad():
            for train_data, _ in data_loader:
                train_data = {key: data.cuda() for key, data in train_data.items()}
                predictions = self.model(train_data)
                predictions = torch.sigmoid(predictions)
                predictions = predictions.cpu().numpy()
                data_dict['index'].extend(list(train_data['index'].cpu().numpy()))
                data_dict['predictions'].extend(list(predictions.flatten()))
                progress_bar.update()
        predict_df = pd.DataFrame.from_dict(data_dict)
        dataset.to_csv(self.dataset_name)
        predict_df.to_csv(self.raw_predictions)
        return predict_df, dataset

    @staticmethod
    def generate_raw_predictions(predictions, gt):
        pred = copy.deepcopy(gt)
        pred['prediction'] = predictions['predictions']
        pred = pred[['patient_id', 'laterality', 'view', 'prediction']]
        pred['prediction_id'] = pred['patient_id'].astype(str) + '_' + pred['laterality']
        pred = pred.groupby(['prediction_id'], as_index=False).agg({'prediction': 'mean'})
        pred = pred.rename(columns={'prediction': 'cancer'})
        return pred

    @staticmethod
    def generate_gt_submission(gt):
        gt_submission = copy.deepcopy(gt)
        gt_submission['prediction_id'] = gt_submission['patient_id'].astype(str) + '_' + gt_submission['laterality']
        gt_submission = gt_submission.groupby(['prediction_id'], as_index=False).agg({'cancer': 'max'})
        return gt_submission

    def calculate_thresholds(self):
        gt = pd.read_csv(self.dataset_name)
        predictions = pd.read_csv(self.raw_predictions)

        raw_predictions = self.generate_raw_predictions(predictions, gt)
        gt_submission = self.generate_gt_submission(gt)

        pf_beta_max = 0
        best_pf_beta_threshold = 0
        for i in range(1, 10):
            pred = copy.deepcopy(raw_predictions)
            pf_beta_threshold = i / 10

            target = torch.from_numpy(gt_submission['cancer'].values)

            binary_accuracy = Accuracy(threshold=pf_beta_threshold)
            binary_precision = Precision(threshold=pf_beta_threshold)
            binary_recall = Recall(threshold=pf_beta_threshold)
            binary_specificity = Specificity(threshold=pf_beta_threshold)
            precision = binary_precision(torch.Tensor(pred['cancer']), target)
            recall = binary_recall(torch.Tensor(pred['cancer']), target)
            accuracy = binary_accuracy(torch.Tensor(pred['cancer']), target)
            specificity = binary_specificity(torch.Tensor(pred['cancer']), target)

            pred['cancer'] = pred['cancer'].apply(lambda x: 1 if x > pf_beta_threshold else 0)
            pf_beta = pfbeta_torch(torch.Tensor(pred['cancer']), torch.Tensor(gt_submission['cancer']))
            print(f" pf_beta_threshold {pf_beta_threshold:.2f} pf_beta {pf_beta:.2f} precision {precision:.2f} recall {recall:.2f} accuracy {accuracy:.4f} specificity {specificity:.4f}")

            if pf_beta > pf_beta_max:
                best_pf_beta_threshold = pf_beta_threshold
                pf_beta_max = pf_beta
        self.threshold = best_pf_beta_threshold
        print(f"best threshold {best_pf_beta_threshold} pf_beta {pf_beta_max}")

    def submission(self, dataset):
        assert self.threshold
        predict_df, dataset_df = self.inference(dataset)
        submission_df = self.generate_raw_predictions(predict_df, dataset_df)
        submission_df['cancer'] = submission_df['cancer'].apply(lambda x: 1 if x > self.threshold else 0)
        submission_df.to_csv("submission.csv")


