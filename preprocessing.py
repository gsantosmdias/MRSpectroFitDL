"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


def interpolate_matrix(original_data, new_shape=(224, 224)):
    if not isinstance(original_data, torch.Tensor):
        original_data = torch.from_numpy(original_data)

    if original_data.is_complex():
        real_part = original_data.real.float()
        imag_part = original_data.imag.float()

        real_part = real_part.unsqueeze(1)
        imag_part = imag_part.unsqueeze(1)

        resized_real = F.interpolate(real_part, size=new_shape, mode='bilinear', align_corners=False)
        resized_imag = F.interpolate(imag_part, size=new_shape, mode='bilinear', align_corners=False)

        resized_data = torch.complex(resized_real, resized_imag)
    else:
        original_data = original_data.float().unsqueeze(1)
        resized_data = F.interpolate(original_data, size=new_shape, mode='bilinear', align_corners=False)

    return resized_data


def torch_z_score_normalize(tensor):
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std_dev = tensor.std(dim=(1, 2), keepdim=True)

    normalized_tensor = (tensor - mean) / std_dev

    return normalized_tensor


class STFT(nn.Module):
    def __init__(self, window_size, hop_size, ppm_lim, t_lim):
        super().__init__()
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.ppm_lim = eval(ppm_lim)
        self.t_lim = eval(t_lim)

    def forward(self, x, fs, larmorfreq):
        window = torch.hann_window(self.window_size, device=x.device)

        Zxx = torch.stft(x, n_fft=self.window_size, hop_length=self.hop_size,
                         win_length=self.window_size, window=window, return_complex=True, onesided=False)

        spec_mag = torch.abs(Zxx)

        f = torch.fft.fftfreq(self.window_size, (1 / fs))
        f = f.unsqueeze(0).expand(Zxx.shape[0], -1)

        num_frames = spec_mag.shape[-1]
        t = torch.linspace(0, (num_frames - 1) * (self.hop_size / fs), num_frames)
        t = t.unsqueeze(0).expand(Zxx.shape[0], -1)

        Zxx_parts = torch.chunk(Zxx, 2, dim=1)
        Zxx_ordered = torch.cat((Zxx_parts[1], Zxx_parts[0]), dim=1)

        f_parts = torch.chunk(f, 2, dim=1)
        f_ordered = torch.cat((f_parts[1], f_parts[0]), dim=1)

        ppm_axis = 4.65 - f_ordered / larmorfreq

        ppm_mask = (ppm_axis >= self.ppm_lim[0]) & (ppm_axis <= self.ppm_lim[1])
        t_mask = (t >= self.t_lim[0]) & (t <= self.t_lim[1])

        ppm_mask_single = ppm_mask[0, :]
        t_mask_single = t_mask[0, :]

        filtered_freq = Zxx_ordered[:, ppm_mask_single, :]

        Zxx_zoomed = filtered_freq[:, :, t_mask_single]

        spectrogram = interpolate_matrix(Zxx_zoomed)
        spectrogram = torch.flip(spectrogram, dims=[2])

        spectrogram_real = spectrogram.real[:, 0, :, :]
        spectrogram_imag = spectrogram.imag[:, 0, :, :]
        spectrogram_mag = torch.abs(spectrogram[:, 0, :, :])

        spectrogram_real = torch_z_score_normalize(spectrogram_real)
        spectrogram_imag = torch_z_score_normalize(spectrogram_imag)
        spectrogram_mag = torch_z_score_normalize(spectrogram_mag)

        three_channels_spectrogram = torch.stack((spectrogram_real, spectrogram_imag, spectrogram_mag), dim=1)

        return three_channels_spectrogram
