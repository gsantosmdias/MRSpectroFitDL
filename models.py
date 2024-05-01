"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import torch.nn as nn
import timm
import torch
from preprocessing import STFT


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size[0], img_size[1]))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out


class TimmAdvancedModelSpectrogram(nn.Module):
    def __init__(self,
                 window_size: int,
                 hop_size: int,
                 ppm_lim: tuple,
                 t_lim: tuple,
                 timm_network: str = "vit_base_patch16_224",
                 image_size: tuple[int, int] = (224, 224),
                 nchannels: int = 3,
                 pretrained: bool = True,
                 num_classes: int = 0,
                 transformers: bool = True,
                 features_only: bool = True
                 ):

        super().__init__()
        if transformers:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.transformers = transformers
        self.encoder = timm.create_model(**model_creator)
        self.stft = STFT(window_size, hop_size, ppm_lim, t_lim)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        self.linear_1 = nn.Linear(n_out, 128)
        self.linear_2 = nn.Linear(128, 24)
        self.relu = nn.ReLU()

    def forward(self, signal_input, fs, larmorfreq):
        spectrogram_3ch = self.stft(signal_input, fs, larmorfreq)

        out_encoder = self.encoder(spectrogram_3ch) if self.transformers else self.encoder(spectrogram_3ch)[-1]

        out2 = self.relu(self.linear_1(out_encoder))

        out3 = self.linear_2(out2)

        return out3


class TimmAdvancedResidualModelSpectrogram(nn.Module):
    def __init__(self,
                 window_size: int,
                 hop_size: int,
                 ppm_lim: tuple,
                 t_lim: tuple,
                 dropout: int,
                 timm_network: str = "vit_base_patch16_224",
                 image_size: tuple[int, int] = (224, 224),
                 nchannels: int = 3,
                 pretrained: bool = True,
                 num_classes: int = 0,
                 transformers: bool = True,
                 features_only: bool = True,
                 kernel_size: int = 2,
                 stride: int = 2):

        super().__init__()
        if transformers:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.transformers = transformers

        self.encoder = timm.create_model(**model_creator)
        self.stft = STFT(window_size, hop_size, ppm_lim, t_lim)

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, count_include_pad=False)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(n_out, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_4 = nn.Linear(128, 24)

    def forward(self, signal_input, fs, larmorfreq):
        spectrogram_3ch = self.stft(signal_input, fs, larmorfreq)

        out_encoder = self.encoder(spectrogram_3ch) if self.transformers else self.encoder(spectrogram_3ch)[-1]

        out1 = self.relu(self.linear_1(out_encoder))
        out1 = self.dropout(out1)

        out_pooled = self.avg_pool(out1.unsqueeze(1)).squeeze(1)

        out2 = self.relu(self.linear_2(out1))
        out2 = self.dropout(out2)

        out2 = out2 + out_pooled

        out3 = self.relu(self.linear_3(out2))
        out3 = self.dropout(out3)

        out4 = self.linear_4(out3)

        return out4


class TimmModelSpectrogram(nn.Module):
    def __init__(self, dropout: int,
                 timm_network: str = "vit_base_patch16_224",
                 image_size: tuple[int, int] = (224, 224),
                 nchannels: int = 3,
                 pretrained: bool = True,
                 num_classes: int = 0,
                 transformers: bool = True,
                 features_only: bool = True,
                 kernel_size: int = 2,
                 stride: int = 2,
                 ):

        super().__init__()
        if transformers:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': timm_network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)
        self.transformers = transformers

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, count_include_pad=False)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(n_out, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, 128)
        self.linear_4 = nn.Linear(128, 24)

    def forward(self, signal_input):
        out_encoder = self.encoder(signal_input) if self.transformers else self.encoder(signal_input)[-1]

        out1 = self.relu(self.linear_1(out_encoder))
        out1 = self.dropout(out1)

        out_pooled = self.avg_pool(out1.unsqueeze(1)).squeeze(1)

        out2 = self.relu(self.linear_2(out1))
        out2 = self.dropout(out2)

        out2 = out2 + out_pooled

        out3 = self.relu(self.linear_3(out2))
        out3 = self.dropout(out3)

        out4 = self.linear_4(out3)

        return out4
