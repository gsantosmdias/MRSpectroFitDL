"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

from losses import MAELoss
from utils import set_device, set_fs_larmorfreq
from models import TimmAdvancedResidualModelSpectrogram, TimmModelSpectrogram, TimmAdvancedModelSpectrogram
from save_models import SaveBestModel
from datasets import DatasetBasisSetOpenTransformer3ChNormalize, DatasetSimpleFID
import torch


save_best_model = SaveBestModel()

DEVICE = set_device()
FS, LARMORFREQ = set_fs_larmorfreq()

FACTORY_DICT = {
    "model": {
        "TimmAdvancedResidualModelSpectrogram": TimmAdvancedResidualModelSpectrogram,
        "TimmModelSpectrogram": TimmModelSpectrogram,
        "TimmAdvancedModelSpectrogram": TimmAdvancedModelSpectrogram

    },
    "dataset": {
        "DatasetBasisSetOpenTransformer3ChNormalize": DatasetBasisSetOpenTransformer3ChNormalize,
        "DatasetSimpleFID": DatasetSimpleFID
    },
    "optimizer": {
        "Adam": torch.optim.Adam
    },
    "loss": {
        "MAELoss": MAELoss,

    },
}
