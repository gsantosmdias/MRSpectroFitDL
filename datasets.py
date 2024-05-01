"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import os
import numpy as np
import torch
from torch_snippets import Dataset
from utils import calculate_spectrogram, get_fid_params, NormalizeData
from basis_and_generator import ReadDataSpectrum


class DatasetBasisSetOpenTransformer3ChNormalize(Dataset):
    def __init__(self, **kargs: dict) -> None:
        self.path_data = kargs['path_data']
        self.list_path_data = sorted(os.listdir(self.path_data))
        self.norm = kargs['norm']
        self.input_path = [pred for pred in self.list_path_data if ".txt" in pred]
        self.output_path = [pred for pred in self.list_path_data if ".json" in pred]
        self.normalizer = NormalizeData()

    def __len__(self) -> int:
        return len(self.list_path_data) // 2

    def __getitem__(self, idx: int) -> (torch.Tensor, np.ndarray):
        self.acess_tmp = f"{self.path_data}/{self.input_path[idx]}"

        sample_for_pred = f"{self.path_data}/{self.input_path[idx]}"
        sample_for_ground_truth = f"{self.path_data}/{self.output_path[idx]}"

        fid_input = ReadDataSpectrum.read_generated(sample_for_pred)
        fid_params = ReadDataSpectrum.ground_truth_json(sample_for_ground_truth)

        spectrogram = calculate_spectrogram(fid_input, 0.00025, hope_size=10)
        rows = [127 - i for i in range(0, 32)]
        spectrogram = np.delete(spectrogram, rows, axis=0)
        spectrogram = np.pad(spectrogram, ((0, 0), (0, 18)), mode='constant', constant_values=0)

        fid_params_array = get_fid_params(fid_params)

        self.snr = fid_params_array[-1]
        fid_params_array = fid_params_array[:-1]

        fid_params_array = torch.from_numpy(fid_params_array)

        spectrogram_real = np.real(spectrogram)
        spectrogram_real = self.normalizer.normalize(spectrogram_real, method=self.norm)
        spectrogram_real = torch.from_numpy(spectrogram_real)

        spectrogram_imag = np.imag(spectrogram)
        spectrogram_imag = self.normalizer.normalize(spectrogram_imag, method=self.norm)
        spectrogram_imag = torch.from_numpy(spectrogram_imag)

        spectrogram_abs = np.abs(spectrogram)
        spectrogram_abs = self.normalizer.normalize(spectrogram_abs, method=self.norm)
        spectrogram_abs = torch.from_numpy(spectrogram_abs)

        spectrogram_real = spectrogram_real.unsqueeze(0)
        spectrogram_imag = spectrogram_imag.unsqueeze(0)
        spectrogram_abs = spectrogram_abs.unsqueeze(0)

        three_channels_spectrogram = torch.cat([spectrogram_real, spectrogram_imag, spectrogram_abs],
                                               dim=0)

        return three_channels_spectrogram.type(torch.FloatTensor), fid_params_array.type(torch.FloatTensor)


class DatasetSimpleFID(Dataset):
    def __init__(self, **kargs: dict) -> None:
        self.path_data = kargs['path_data']
        self.list_path_data = sorted(os.listdir(self.path_data))
        self.input_path = [pred for pred in self.list_path_data if ".txt" in pred]
        self.output_path = [pred for pred in self.list_path_data if ".json" in pred]

    def __len__(self) -> int:
        return len(self.list_path_data) // 2

    def __getitem__(self, idx: int) -> (torch.Tensor, np.ndarray):
        self.acess_tmp = f"{self.path_data}/{self.input_path[idx]}"

        sample_for_pred = f"{self.path_data}/{self.input_path[idx]}"
        sample_for_ground_truth = f"{self.path_data}/{self.output_path[idx]}"

        fid_input = ReadDataSpectrum.read_generated(sample_for_pred)
        fid_params = ReadDataSpectrum.ground_truth_json(sample_for_ground_truth)

        fid_params_array = get_fid_params(fid_params)

        self.snr = fid_params_array[-1]
        fid_params_array = fid_params_array[:-1]

        fid_params_array = torch.from_numpy(fid_params_array)
        fid_input = torch.from_numpy(fid_input)

        return fid_input, fid_params_array.type(torch.FloatTensor)
