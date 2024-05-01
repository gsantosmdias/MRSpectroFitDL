import time
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from statistics import mean
from utils import read_yaml, calculate_fqn, clean_directory
from plot_metrics import PlotMetrics
from basis_and_generator import signal_reconstruction
from constants import *
from basis_and_generator import transform_frequency, ReadDataSpectrum
from torch.utils.data import DataLoader


EPSILON = 1e-12
OUTPUT_LABELS = ['A_Ala', 'A_Asc', 'A_Asp', 'A_Cr', 'A_GABA', 'A_Glc', 'A_Gln', 'A_Glu',
                 'A_Gly', 'A_GPC', 'A_GSH', 'A_Ins', 'A_Lac', 'A_MM', 'A_NAA', 'A_NAAG',
                 'A_PCho', 'A_PCr', 'A_PE', 'A_sIns', 'A_Tau', 'damping', 'freq_s', 'phase_s']


def calculate_means(metrics, keys):
    return {key: mean(metrics[key]) for key in keys}


def load_model(model_configs, load_dict):
    if isinstance(model_configs, dict):
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](**model_configs[list(model_configs.keys())[0]])
    else:
        model = FACTORY_DICT["model"][model_configs]()
    model.load_state_dict(load_dict["model_state_dict"])
    return model


def load_dataset(dataset_configs):
    test_dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]])

    return test_dataset


def setup_directories(save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)
    clean_directory(save_dir_path)
    for sub_folder in ["input_ground", "pred_ground", "fit_eval", "metrics"]:
        os.makedirs(os.path.join(save_dir_path, sub_folder), exist_ok=True)


def initialize_metrics():
    return {
        "mse": [], "mae": [], "mape": [], "r2": [], "fqn": [],
        "coefs_mse": np.empty((0, 24)), "coefs_mae": np.empty((0, 24)), "coefs_mape": np.empty((0, 24)),
        "coefs_pred": [], "coefs_ground": []
    }


def normalize_and_extend(data):
    amplitude = data[:21]
    min_val, max_val = amplitude.min(), amplitude.max()
    amplitude_norm = (amplitude - min_val) / (max_val - min_val)
    return np.concatenate((amplitude_norm, data[21:]))


def print_summary(processing_time):
    print()
    print(f"The average time computation per sample is: {np.mean(processing_time) * 1000:.2f} ms")


def process_sample(spectrogram_3ch, labels, model, input_spec, filename, basis_set_path, dict_metrics,
                   processing_time, save_dir_path):
    spectrogram_3ch = spectrogram_3ch.to(DEVICE)

    t1 = time.time()

    if list(configs["model"].keys())[0] == 'TimmModelSpectrogram':
        pred_labels = model(spectrogram_3ch)
    else:
        pred_labels = model(spectrogram_3ch, FS, LARMORFREQ)

    t2 = time.time()

    processing_time.append(t2 - t1)

    pred_labels = pred_labels.detach().cpu().numpy().squeeze()
    pred_labels_norm = normalize_and_extend(pred_labels)

    labels = labels.numpy().squeeze()
    labels_norm = normalize_and_extend(labels)

    pred_fid = signal_reconstruction(basis_set_path, list(pred_labels))
    truth_fid = signal_reconstruction(basis_set_path, list(labels))

    pred, ground, ppm_axis = transform_frequency(pred_fid, truth_fid)
    fit_residual = pred - input_spec
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min())
    ground_norm = (ground - ground.min()) / (ground.max() - ground.min())

    mse = np.square(np.real(pred_norm) - np.real(ground_norm)).mean()
    mae = np.abs(np.real(pred_norm) - np.real(ground_norm)).mean()
    mape = np.abs((np.real(ground) - np.real(pred)) / (np.real(ground) + EPSILON)).mean()
    fqn = calculate_fqn(input_spec, fit_residual, ppm_axis)
    r2 = r2_score(np.real(ground), np.real(pred))

    c_mse = np.square(pred_labels_norm - labels_norm)

    c_mae = np.abs(pred_labels_norm - labels_norm)

    c_mape = np.abs((pred_labels - labels) / (labels + EPSILON))

    print(f" \n {filename} results:")
    dict_metrics["mse"].append(mse)
    print(f'MSE Fitting error: {mse}')
    dict_metrics["mae"].append(mae)
    print(f'MAE Fitting error: {mae}')
    dict_metrics["mape"].append(mape)
    print(f'MAPE Fitting error: {mape}')
    dict_metrics["fqn"].append(fqn)
    print(f'FQN Fitting error: {fqn}')
    dict_metrics["coefs_pred"].append(pred_labels)
    dict_metrics["coefs_ground"].append(labels)
    dict_metrics["r2"].append(r2)
    print(f'R2 Fitting error: {r2}')

    dict_metrics["coefs_mse"] = np.vstack((dict_metrics["coefs_mse"], c_mse))
    dict_metrics["coefs_mae"] = np.vstack((dict_metrics["coefs_mae"], c_mae))
    dict_metrics["coefs_mape"] = np.vstack((dict_metrics["coefs_mape"], c_mape))

    PlotMetrics.spectrum_comparison(input_spec, ground, ppm_axis,
                                    label_1="input_spec",
                                    label_2="ground_truth",
                                    fig_name=f"{save_dir_path}/input_ground/{filename}.png")

    PlotMetrics.spectrum_comparison(pred, ground, ppm_axis,
                                    label_1="pred",
                                    label_2="ground_truth",
                                    fig_name=f"{save_dir_path}/pred_ground/{filename}.png")

    PlotMetrics.plot_fitted_evaluation(input_spec, pred, fit_residual, (ground - pred), ppm_axis,
                                       f"{save_dir_path}/fit_eval/{filename}.png")

    return dict_metrics, processing_time


def evaluate_test_data(test_dataset, model, basis_set_path, save_dir_path):
    dict_metrics = initialize_metrics()
    processing_time = []

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, dataset in enumerate(tqdm(test_loader)):
        spectrogram_3ch, labels = dataset

        input_spec = ReadDataSpectrum.load_txt_spectrum(f"{test_dataset.path_data}/{test_dataset.input_path[i]}")

        filename = test_dataset.input_path[i].split(".")[0]

        dict_metrics, processing_time = process_sample(spectrogram_3ch, labels, model, input_spec, filename,
                                                        basis_set_path, dict_metrics, processing_time, save_dir_path)

        if i == 0:
            dict_metrics["coefs_mse"] = dict_metrics["coefs_mse"][1:, :]
            dict_metrics["coefs_mae"] = dict_metrics["coefs_mae"][1:, :]
            dict_metrics["coefs_mape"] = dict_metrics["coefs_mape"][1:, :]

    print_summary(processing_time)

    mean_mse = mean(dict_metrics["mse"])
    mean_mae = mean(dict_metrics["mae"])
    mean_mape = mean(dict_metrics["mape"])
    mean_fqn = mean(dict_metrics["fqn"])
    mean_r2 = mean(dict_metrics["r2"])

    mean_mse_coeff = np.mean(dict_metrics["coefs_mse"], axis=0)
    mean_mae_coeff = np.mean(dict_metrics["coefs_mae"], axis=0)
    mean_mape_coeff = np.mean(dict_metrics["coefs_mape"], axis=0)

    r2_scores = r2_score(np.array(dict_metrics["coefs_ground"]), np.array(dict_metrics["coefs_pred"]),
                         multioutput='raw_values')

    df_mean_fitting_metrics = pd.DataFrame(data=[[mean_mse, mean_mae, mean_mape, mean_fqn, mean_r2]],
                                           columns=["MSE", "MAE", "MAPE", "FQN", "R2"])
    df_mean_fitting_metrics["MAPE"] = df_mean_fitting_metrics["MAPE"] * 100

    df_mse_coeff = pd.DataFrame(data=mean_mse_coeff[np.newaxis, :],
                                columns=OUTPUT_LABELS,
                                index=[["MSE"]])
    df_mae_coeff = pd.DataFrame(data=mean_mae_coeff[np.newaxis, :],
                                columns=OUTPUT_LABELS,
                                index=[["MAE"]])
    df_mape_coeff = pd.DataFrame(data=100 * mean_mape_coeff[np.newaxis, :],
                                 columns=OUTPUT_LABELS,
                                 index=[["MAPE"]])
    df_r2_coeff = pd.DataFrame(data=r2_scores[np.newaxis, :],
                               columns=OUTPUT_LABELS,
                               index=[["R2"]])

    df_coef_metrics = pd.concat([df_mse_coeff, df_mae_coeff, df_mape_coeff, df_r2_coeff])

    df_mean_fitting_metrics.to_csv(f"{save_dir_path}/metrics/df_mean_fitting_metrics.csv")
    df_coef_metrics.to_csv(f"{save_dir_path}/metrics/coefficient_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config_file", type=str, help="Configuration file path")
    parser.add_argument("weight", type=str, help="Weights for the neural network")
    args = parser.parse_args()

    configs = read_yaml(args.config_file)
    load_dict = torch.load(args.weight)
    model = load_model(configs['model'], load_dict)
    model.to(DEVICE)
    model.eval()

    save_dir_path = "evaluate_results"
    setup_directories(save_dir_path)

    test_dataset = load_dataset(configs["test_dataset"])

    evaluate_test_data(test_dataset, model, configs["basis_set_path"], save_dir_path)
