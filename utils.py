"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import torch
import yaml
import numpy as np
from typing import List
import h5py
from scipy import signal
import os


def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print('Using {}'.format(device))

    return device


def set_fs_larmorfreq():
    fs = 4000
    larmorfreq = 123.2

    return fs, larmorfreq


def clean_directory(dir_path):
    for file_name in os.listdir(dir_path):
        file_absolute_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_absolute_path):
            os.remove(file_absolute_path)
        elif os.path.isdir(file_absolute_path):
            clean_directory(file_absolute_path)
            os.rmdir(file_absolute_path)


def read_yaml(file: str) -> yaml.loader.FullLoader:
    with open(file, "r") as yaml_file:
        configurations = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return configurations


class ReadDatasets:
    @staticmethod
    def read_h5_pred_fit(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with h5py.File(filename) as hf:
            input_spec = hf["input_spec"][()]
            ground = hf["ground"][()]
            pred = hf["pred"][()]
            ppm = hf["ppm"][()]

        return input_spec, ground, pred, ppm

    @staticmethod
    def write_h5_pred_fit(save_file_path: str,
                          input_spec: np.ndarray,
                          ground: np.ndarray,
                          pred: np.ndarray,
                          ppm: np.ndarray) -> None:
        with h5py.File(save_file_path, 'w') as hf:
            hf.create_dataset('input_spec', data=input_spec)
            hf.create_dataset('ground', data=ground)
            hf.create_dataset('pred', data=pred)
            hf.create_dataset('ppm', data=ppm)

    @staticmethod
    def read_h5(filename: str) -> List[np.ndarray]:
        with h5py.File(filename, "r") as f:
            a_group_key = list(f.keys())[0]

            data = list(f[a_group_key])

            return data


def get_fid_params(params: dict) -> np.ndarray:
    params.pop("A_Ace", None)
    params_array = []
    for key in params.keys():
        params_array.append(params[key])

    return np.asarray(params_array)


def calculate_spectrogram(FID, t, window_size=256, hope_size=64, window='hann', nfft=None):
    noverlap = window_size - hope_size

    if not signal.check_NOLA(window, window_size, noverlap):
        raise ValueError("signal windowing fails Non-zero Overlap Add (NOLA) criterion; "
                         "STFT not invertible")
    fs = 1 / t
    _, _, Zxx = signal.stft(FID, fs=fs, nperseg=window_size, noverlap=noverlap,
                            return_onesided=False, nfft=nfft)
    return Zxx


def calculate_fqn(spec, residual, ppm):
    dt_max_ind, dt_min_ind = np.amax(np.where(ppm >= 9.8)), np.amin(np.where(ppm <= 10.8))
    noise_var = np.var(spec[dt_min_ind:dt_max_ind])
    residual_var = np.var(residual)

    fqn = residual_var / noise_var
    return fqn


class NormalizeData:
    def normalize(self, arr, method):
        if method == "min-max":
            return self.min_max_normalize(arr)
        elif method == "z_norm":
            return self.z_score_normalize(arr)

    def min_max_normalize(self, arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    def z_score_normalize(self, arr):
        mean = np.mean(arr)
        std_dev = np.std(arr)
        normalized_arr = (arr - mean) / std_dev
        return normalized_arr
