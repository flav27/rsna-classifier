from typing import Optional, Union, List, Any, Tuple

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


def pfbeta_torch(preds, labels, beta=1):
    """
    Official metric of RSNA Screening Mammography Breast Cancer Detection
    This extension of the traditional F score accepts probabilities instead of binary classifications
    """
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0


class PFBeta(Metric):

    """
    subclass of torchmetrics.Metrics that uses probabilistic f1 score
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = False

    def __init__(
            self,
            ignore_index: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        return pfbeta_torch(dim_zero_cat(self.preds), dim_zero_cat(self.target))
