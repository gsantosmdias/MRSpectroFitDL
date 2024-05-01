"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import NormalizeData


class PlotMetrics:
    @staticmethod
    def spectrum_comparison(spec_1, spec_2, ppm,
                            label_1, label_2, fig_name,):

        normalizer = NormalizeData()
        spec_1 = normalizer.min_max_normalize(spec_1)
        spec_2 = normalizer.min_max_normalize(spec_2)

        fig, ax = plt.subplots()

        ax.plot(ppm, np.real(spec_1), label=label_1, color="black")
        ax.plot(ppm, np.real(spec_2), label=label_2, color="red")

        plt.xlim((5, 0))
        plt.grid()
        plt.legend()
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def plot_fitted_evaluation(data_signal, fitted_signal, fit_residual_signal, ground_truth_minus_fit, ppm_axis,
                               fig_name, spacing=1e4):

        data_signal = np.real(data_signal)
        fitted_signal = np.real(fitted_signal)
        fit_residual_signal = np.real(fit_residual_signal)
        ground_truth_minus_fit = np.real(ground_truth_minus_fit)

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.plot(ppm_axis, data_signal, label="Input", color="black")
        ax.plot(ppm_axis, fitted_signal + 1.2 * spacing, label="Fit", color="red")
        ax.plot(ppm_axis, fit_residual_signal + 3.5 * spacing, label="Fit Residual", color="orange")
        ax.plot(ppm_axis, ground_truth_minus_fit + 3.8 * spacing, label="Ground Truth minus Fit", color="green")

        ax.set_xlabel("Chemical Shift (ppm)")
        ax.yaxis.set_visible(False)
        ax.set_title("Fitting Evaluation")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        plt.tight_layout()
        plt.xlim((5, 0))
        plt.savefig(fig_name)
        plt.close(fig)

    @staticmethod
    def plot_fitted_evaluation_beta(data_signal, fitted_signal, fit_residual_signal, ground_truth_minus_fit, ppm_axis,
                                    fig_name, spacing=1e4):
        data_signal = np.real(data_signal)
        fitted_signal = np.real(fitted_signal)
        fit_residual_signal = np.real(fit_residual_signal)
        ground_truth_minus_fit = np.real(ground_truth_minus_fit)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(ppm_axis, data_signal, label="Input", color="black")
        ax.plot(ppm_axis, fitted_signal + 1.2 * spacing, label="Fit", color="red")
        ax.plot(ppm_axis, fit_residual_signal + 3.5 * spacing, label="Fit Residual", color="orange")
        ax.plot(ppm_axis, ground_truth_minus_fit + 3.8 * spacing, label="Ground Truth minus Fit", color="green")

        ax.set_xlabel("Chemical Shift (ppm)")
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['bottom'].set_linewidth(0.5)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        plt.tight_layout()
        plt.xlim((5, 0))

        plt.savefig(fig_name, transparent=True, bbox_inches='tight')
        plt.close(fig)
