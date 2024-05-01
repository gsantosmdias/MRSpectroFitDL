"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import numpy as np
import json


class ReadDataSpectrum:
    @staticmethod
    def read_generated(path_name: str):
        return np.loadtxt(path_name, dtype=np.complex128)

    @staticmethod
    def ground_truth_json(path_name: str):
        with open(path_name, 'r') as param_json:
            params = json.load(param_json)
        return params

    @staticmethod
    def load_txt_spectrum(txt_path):
        fid = ReadDataSpectrum.read_generated(txt_path)
        spec = np.fft.fftshift(np.fft.fft(fid))
        return spec


class BasisRead:
    def __init__(self, basis_path, sample_time=0.00025):
        self.basis_path = basis_path
        self.basis = self.text_to_array_basis()
        self.fids = self.basis
        self.noise = np.zeros(len(self.fids[0]), dtype="complex_")
        self.period = sample_time

    def text_to_array_basis(self) -> np.ndarray:
        metab_list = ['Ace.txt', 'Ala.txt', 'Asc.txt', 'Asp.txt', 'Cr.txt', 'GABA.txt', 'Glc.txt', 'Gln.txt', 'Glu.txt',
                      'Gly.txt', 'GPC.txt', 'GSH.txt', 'Ins.txt', 'Lac.txt', 'Mac.txt', 'NAA.txt', 'NAAG.txt',
                      'PCho.txt', 'PCr.txt', 'PE.txt', 'sIns.txt', 'Tau.txt']
        basis_list = []
        for metab in metab_list:
            metab_path = self.basis_path + "/" + metab

            with open(metab_path, 'r') as file:
                lines = file.readlines()

            met_value = []
            for line in lines:
                values = line.strip().split()
                value = complex(float(values[0]), float(values[1]))
                met_value.append(value)
            basis_list.append(met_value)
        return np.asarray(basis_list)

    def scale(self, scale_factors):
        self.fids = np.asarray([basis * scale_factor for basis, scale_factor in zip(self.basis, scale_factors)])

    def damp_and_shift(self, factors):
        start = 0
        stop = len(self.fids[0]) * self.period
        time_range = np.linspace(start, stop, num=len(self.fids[0]))
        terms = []

        for t in time_range:
            terms.append(np.exp((factors[0] + 2 * np.pi * factors[1] * 1j) * t) * np.exp(factors[2] * 1j))
        terms = np.asarray(terms)

        for i in range(len(self.fids)):
            self.fids[i] = self.fids[i] * terms

    def noise_by_SNR(self, snr):
        self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        for fid in self.fids:
            self.r_fid += fid
        std_deviation = np.amax(np.abs(np.fft.fftshift(np.fft.fft(self.r_fid)))) / snr
        noise_temp = np.random.normal(0, np.sqrt(2.0) * std_deviation / 2.0, size=(len(self.r_fid), 2)).view(complex)
        self.noise = []
        for point in noise_temp:
            self.noise.append(point[0])
        self.noise = np.asarray(self.noise)
        self.noise = np.fft.ifftshift(np.fft.ifft(self.noise))

    def gen_resulting_fid(self):
        self.r_fid = np.zeros(len(self.fids[0]), dtype="complex_")
        for fid in self.fids:
            self.r_fid += fid
        self.r_fid += self.noise

        return self.r_fid

    def reset_fid(self):
        self.fids = self.basis


def signal_reconstruction(basis_path, params):
    basis = BasisRead(basis_path)
    basis.scale(([0] + params)[:22])
    basis.damp_and_shift(([0] + params)[22:])
    reconstructed_fid = basis.gen_resulting_fid()
    return reconstructed_fid


def transform_frequency(pred_fid, ground_fid):
    frequency_axis = np.linspace(2000, -2000, 2048)

    ppm_axis = 4.65 + frequency_axis / 123.22

    pred_spec = np.fft.fftshift(np.fft.fft(pred_fid))
    truth_spec = np.fft.fftshift(np.fft.fft(ground_fid))
    return pred_spec, truth_spec, ppm_axis


