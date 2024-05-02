[![gsantosmdias](https://img.shields.io/badge/gsantosmdias-MRSpectroFitDL-blue)](https://github.com/gsantosmdias/MRSpectroFitDL)


# Improving Magnetic Resonance Spectroscopy Fitting with Vision Transformers and Spectrograms

## Introduction

- **Magnetic Resonance Spectroscopy (MRS)**: Offers a non-invasive approach for studying the biochemical composition of brain tissue [1].
- **Enhancing MRS Analysis with Deep Learning (DL)**: Offers an alternative to traditional MRS fitting methods with improved handling of noisy signals, quantification of low-concentration metabolites, and differentiation of overlapping metabolites [2].

**Goal**: Refine Almeida M et al. DL MRS fitting model [![MICLab](https://img.shields.io/badge/MICLab-Spectral%20Fitting%20SIPAIM-red)](https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM) [3] by optimizing the model's architecture, improving spectrogram processing with added temporal and frequency filtering, interpolation, and preprocessing optimization through vectorization and GPU handling.

## Materials and Methods

- **Dataset**: 13128 synthetic MRS signals, short-echo time PRESS sequence, 20 metabolites, and macromolecule contributions.

- **Parameter Prediction**: Predicted 24 parameters, covering metabolite amplitudes, macromolecule contributions, and signal characteristics (global damping, frequency, phase shift).

- **Enhanced Spectrogram Processing**: Utilized temporal filtering on the initial 60% to reduce noise, applied frequency filtering from 0 to 6 ppm for main metabolites' information, and employed bilinear interpolation for upsampling spectrograms, as well as vectorization and GPU handling.

- **Model Architecture**: Three-channel spectrograms (real, imaginary, and magnitude components) are fed into a pre-trained Vision Transformer [4] encoder, whose embedding is decoded for the fitting parameters using fully connected layers. (Fig. 1).
![diagram_sipaim-Página-2 (2)](https://github.com/gsantosmdias/MRSpectroFitDL/assets/91618118/0c514950-78e8-4b1e-add8-301c1c7ee82a)

*Figure 1: Model architecture.*

## Results
- Evaluated using 128 test samples.
- The results demonstrate that the enhanced model achieves superior fitting accuracy across several key metabolites and parameters.
- The model outperformed those of Almeida M et al. and Shamaei A et al. [![amirshamaei](https://img.shields.io/badge/amirshamaei-Deep--MRS--Quantification-brightgreen)](https://github.com/amirshamaei/Deep-MRS-Quantification) [5] across 15 parameters for Coefficient of Determination (R²) and 12 parameters for Mean Percentage Error (MAPE).

## Discussion/Conclusion

**Key Insights**: Spectrograms boost DL fitting performance. This study outperformed Almeida M et al. approach, highlighting that accurate processing yields superior MRS fitting features.

**Next Steps**: Upcoming studies to test on in-vivo data and benchmark against  well-known MRS software, e.g., LCModel [6].

## Conference Presentation
This project was presented at the [10th BRAINN Congress](https://www.brainncongress.com/10th-brainn-congress-2024/), showcasing the developments in MRS fitting techniques.

## Acknowledgments
This work is an extension of the foundational research and open-source repository [![MICLab](https://img.shields.io/badge/MICLab-Spectral%20Fitting%20SIPAIM-red)](https://github.com/MICLab-Unicamp/Spectral_fitting_SIPAIM), which has has adapted the robust and flexible Spectro-ViT training framework [![GitHub](https://img.shields.io/badge/MICLab-Spectro_ViT-purple)](https://github.com/MICLab-Unicamp/Spectro-ViT).

## References
[1] Gujar S et al., [doi:10.1097/ 01.wno.0000177307.21081.81](https://journals.lww.com/jneuro-ophthalmology/fulltext/2005/09000/Magnetic_Resonance_Spectroscopy.00015.aspx?casa_token=6JXnmc_2ecEAAAAA:rsGgZMLexx7Kkylxbb_2sAJC2jTm-wfI9kqY_Bi5C23Wloqi0oBIaplGr8M_DOmKH6IYMRwO7XupAlajOsMKDv1re0FHjAW9pF8);

[2] Sande D et al., [doi:10.1002/mrm.29793](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29793); 

[3] Almeida M et al., [doi:10.1109/SIPAIM56729.2023.10373415](https://ieeexplore.ieee.org/abstract/document/10373415?casa_token=Ahsuzc-Og0wAAAAA:K280-T4VONeWQABDT9YJ3tYaGMOhfNOpQswSHgt6GF-7MTAoKDAS6fr9D__R9w5GuLvB6wLI5mk); 

[4] Dosovitskiy A et al., [doi:10.48550/arXiv.2010.11929](https://arxiv.org/abs/2010.11929);

[5] Shamaei A et al., [doi:10.1016/j.compbiomed.2023.106837](https://www.sciencedirect.com/science/article/pii/S0010482523003025); 

[6] Provencher SW et al., [doi:10.1002/nbm.698](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/nbm.698).


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gsantosmdias/MRSpectroFitDL.git
   ```
2. Navigate to the project directory:

   ```bash
   cd MRSpectroFitDL
   ```
3. Check the Python version `3.9.16` in requirements.txt and install the required dependencies:

    ```bash
   pip install -r requirements.txt
   ```
## Training Configuration File Details

The model's training and evaluation behaviors are fully configurable through a YAML configuration file. Below, you will find detailed explanations of key sections within this file:

### Weights & Biases (wandb) Configuration

- `activate`: Enables or disables integration with Weights & Biases. Set to `True` for tracking and visualizing metrics during training.
- `project`: Specifies the project name under which the runs should be logged in Weights & Biases.
- `entity`: Specifies the user or team name under Weights & Biases where the project is located.

### Saving/Reloading Configuration

- **Current Model**
  - `save_model`: Enable or disable the saving of model weights.
  - `model_dir`: Directory to save the model weights.
  - `model_name`: Name under which to save the model weights.

- **Reload from Existing Model**
  - `activate`: Enable reloading weights from a previously saved model to continue training.
  - `model_dir`: Directory from where the model weights should be loaded.
  - `model_name`: Name of the model weights file to be loaded.

### Model Configuration
- **Model**
  - `Model Class Name`: Specifies the model class. Example: `TimmAdvancedModelSpectrogram`.
    - `Instantiation parameters of the model class`.

### Training Parameters

- `epochs`: Number of training epochs.
- `optimizer`: Configuration for the optimizer. Example:
  - `Adam`: Specifies using the Adam optimizer.
    - `lr`: Learning rate for the optimizer.

- `loss`: Specifies the loss function used for training. Example: `MAELoss`.

- `lr_scheduler`: Configuration for the learning rate scheduler. Example:
  - `activate`: Enable or disable the learning rate scheduler.
  - `scheduler_type`: Type of scheduler, e.g., `cosineannealinglr`.
  - `info`: Parameters for the scheduler, such as `T_max` (maximum number of iterations) and `eta_min` (minimum learning rate).

### Dataset Configuration

The following configuration parameters are designed to instantiate the Dataset class:

- **Training Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the training dataset. Example: `DatasetSimpleFID`
    - `path_data`: Directory containing the training data.

- **Validation Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the validation dataset. Example: `DatasetSimpleFID`
    - `path_data`: Directory containing the validation data.

- **Valid on the Fly**
  - `activate`: Enables saving plots that help analyze the model's performance on validation data during training.
  - `save_dir_path`: Directory to save these plots.

- **Test Dataset**
  - `Dataset Class Name`: Specifies the class used for managing the test dataset. Example: `DatasetSimpleFID`
    - `path_data`: Directory containing the validation data.
    - `norm`: Normalization to be applied to the spectrogram channels.
    - 
## Training Example 

Here’s an example of how you might train the model using the configuration file provided for the research developed in this study:

```bash
python main.py configs/config_model.yaml
```

## Evaluation Example

Here's an example of how you might evaluate the trained model:

```bash
python evaluate.py configs/config_model.yaml models_h5/TimmModelSpectrogram.pt
```
## Developer

- [Gabriel Dias](https://github.com/gsantosmdias)

## Citation

If you use our model and code in your research please cite:

    @inproceedings{dias2024improving,
      title={Improving Magnetic Resonance Spectroscopy Fitting with Vision Transformers and Spectrograms through Enhanced Localized Processing},
      author={Dias, G. and Oliveira, M. and Almeida, M. and Dertkigil, S. and Rittner, L.},
      booktitle={10th BRAINN Congress},
      year={2023},
      organization={Medical Image Computing Lab - MICLab, UNICAMP}
    }

