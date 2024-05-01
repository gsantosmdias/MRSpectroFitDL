"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
"""

import argparse
import gc
import os
import random
import shutil
import wandb
from tqdm import trange
from constants import *
from lr_scheduler import CustomLRScheduler
from utils import read_yaml, clean_directory
from metrics import calculate_metrics
from plot_metrics import PlotMetrics
from basis_and_generator import transform_frequency, signal_reconstruction
from basis_and_generator import ReadDataSpectrum

plot_metrics = PlotMetrics()


def valid_on_the_fly(model, epoch, configs, basis_set_path, save_dir_path):
    model.eval()

    val_dataset_configs = configs["valid_dataset"]
    val_dataset = FACTORY_DICT["dataset"][list(val_dataset_configs)[0]](
        **val_dataset_configs[list(val_dataset_configs.keys())[0]])

    random_index = random.randint(0, len(val_dataset) - 1)

    val_path_dataset = val_dataset_configs[list(val_dataset_configs.keys())[0]]['path_data']

    list_val_data = sorted(os.listdir(val_path_dataset))
    list_val_data = [os.path.join(val_path_dataset, file) for file in list_val_data]

    input_spec = ReadDataSpectrum.load_txt_spectrum([fid_txt for fid_txt in list_val_data if ".txt" in fid_txt][random_index])

    spectrogram_3ch, labels = val_dataset[random_index]
    spectrogram_3ch = torch.unsqueeze(spectrogram_3ch, dim=0).to("cuda")

    if list(configs["model"].keys())[0] == 'TimmModelSpectrogram':
        prediction = model(spectrogram_3ch)
    else:
        prediction = model(spectrogram_3ch, FS, LARMORFREQ)

    prediction = prediction.detach().cpu().numpy().squeeze()
    labels = labels.detach().cpu().numpy()

    pred_fid = signal_reconstruction(basis_set_path, list(prediction))
    truth_fid = signal_reconstruction(basis_set_path, list(labels))

    pred_spec, ground_spec, ppm = transform_frequency(pred_fid, truth_fid)

    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(os.path.join(save_dir_path, "input_ground"), exist_ok=True)
    os.makedirs(os.path.join(save_dir_path, "pred_ground"), exist_ok=True)

    PlotMetrics.spectrum_comparison(input_spec, ground_spec, ppm,
                                    label_1="input_spec",
                                    label_2="ground_truth",
                                    fig_name=f"{save_dir_path}/input_ground/epoch_{epoch + 1}.png")

    PlotMetrics.spectrum_comparison(pred_spec, ground_spec, ppm,
                                    label_1="pred",
                                    label_2="ground_truth",
                                    fig_name=f"{save_dir_path}/pred_ground/epoch_{epoch + 1}.png")


class ToolsWandb:
    @staticmethod
    def config_flatten(config, parent_key='', sep='_'):
        items = []
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(ToolsWandb.config_flatten(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)


def get_dataset(dataset_configs):
    dataset = FACTORY_DICT["dataset"][list(dataset_configs)[0]](
        **dataset_configs[list(dataset_configs.keys())[0]]
    )

    return dataset


def set_samples_dataset(configs, samples, type_dataset='train_dataset', key_data="path_data"):
    configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = samples
    return configs


def set_length_dataset(configs, len_, type_dataset='train_dataset', key_data="length_dataset"):
    configs[type_dataset][list(configs[type_dataset].keys())[0]][key_data] = len_
    return configs


def experiment_factory(configs):
    train_dataset_configs = configs["train_dataset"]
    train_dataset_key = list(configs["train_dataset"].keys())[0]

    validation_dataset_configs = configs["valid_dataset"]
    validation_dataset_key = list(configs["valid_dataset"].keys())[0]

    if (not isinstance(train_dataset_configs[train_dataset_key]["path_data"], list)) and (
            not isinstance(validation_dataset_configs[validation_dataset_key]["path_data"], list)):
        print(f"length train: {len(os.listdir(train_dataset_configs[train_dataset_key]['path_data'])) // 2}")
        print(
            f"length validation: {len(os.listdir(validation_dataset_configs[validation_dataset_key]['path_data'])) // 2}")

    model_configs = configs["model"]
    optimizer_configs = configs["optimizer"]
    criterion_configs = configs["loss"]

    train_dataset = get_dataset(train_dataset_configs)
    validation_dataset = get_dataset(validation_dataset_configs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs["train"]["batch_size"], shuffle=True,
        num_workers=configs["train"]["num_workers"]
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=configs["valid"]["batch_size"], shuffle=False,
        num_workers=configs["valid"]["num_workers"]
    )

    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    optimizer = FACTORY_DICT["optimizer"][list(optimizer_configs.keys())[0]](
        model.parameters(), **optimizer_configs[list(optimizer_configs.keys())[0]]
    )

    if type(criterion_configs) == dict:
        criterion = FACTORY_DICT["loss"][list(criterion_configs.keys())[0]](
            **criterion_configs[list(criterion_configs.keys())[0]]
        )
    else:
        criterion = FACTORY_DICT["loss"][criterion_configs]()

    return model, train_loader, validation_loader, optimizer, \
        criterion


def run_train_epoch(model, optimizer, criterion, loader,
                    epoch):
    model.to(DEVICE)
    model.train()

    running_loss = 0
    running_mae = 0
    running_mse = 0

    with trange(len(loader), desc='Train Loop') as progress_bar:
        for batch_idx, sample_batch in zip(progress_bar, loader):
            optimizer.zero_grad()

            input, labels = sample_batch[0], sample_batch[1]

            input = input.to(DEVICE)
            labels = labels.to(DEVICE)

            if list(configs["model"].keys())[0] == 'TimmModelSpectrogram':
                prediction = model(input)
            else:
                prediction = model(input, FS, LARMORFREQ)

            loss = criterion(prediction, labels)

            result = calculate_metrics(prediction, labels)

            running_mse += result['mse']
            running_mae += result['mae']
            running_loss += loss.item()

            progress_bar.set_postfix(
                desc=(f'[epoch: {epoch + 1:d}], iteration: {batch_idx:d}/{len(train_loader):d}, '
                      f'loss: {running_loss / (batch_idx + 1)} | '
                      f"MSE:{running_mse / (batch_idx + 1):.7f} | "
                      f"MAE:{running_mae / (batch_idx + 1):.7f}"
                      )
            )

            loss.backward()
            optimizer.step()

            if configs['wandb']["activate"]:
                wandb.log({'train_loss': loss})

        running_loss = (running_loss / len(loader))

    return running_loss


def run_validation(model, criterion, loader,
                   epoch, configs):
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

        model.to(DEVICE)
        model.eval()

        running_loss = 0
        running_mae = 0
        running_mse = 0

        with trange(len(loader), desc='Validation Loop') as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, loader):
                input, labels = sample_batch[0], sample_batch[1]

                input = input.to(DEVICE)
                labels = labels.to(DEVICE)

                if list(configs["model"].keys())[0] == 'TimmModelSpectrogram':
                    prediction = model(input)
                else:
                    prediction = model(input, FS, LARMORFREQ)

                loss = criterion(prediction, labels)

                result = calculate_metrics(prediction, labels)

                running_mse += result['mse']
                running_mae += result['mae']
                running_loss += loss.item()

                progress_bar.set_postfix(

                    desc=(f"[Epoch {epoch + 1}] Loss: {running_loss / (batch_idx + 1)} | "
                          f"MSE:{running_mse / (batch_idx + 1):.7f} | "
                          f"MAE:{running_mae / (batch_idx + 1):.7f}"
                          ))

    loader_loss = (running_loss / len(loader))
    loader_mean_mse = running_mse / len(loader)
    loader_mean_mae = running_mae / len(loader)

    if configs['wandb']["activate"]:
        wandb.log({'mean_valid_loss': loss})
        wandb.log({'mean_mae': loader_mean_mae})
        wandb.log({'mean_mse': loader_mean_mse})

    if configs['current_model']['save_model']:
        save_path_model = f"{configs['current_model']['model_dir']}/{configs['current_model']['model_name']}.pt"
        save_best_model(loader_loss, model, save_path_model)

    if configs["valid_on_the_fly"]["activate"]:
        valid_on_the_fly(model, epoch, configs, configs["basis_set_path"], configs["valid_on_the_fly"]["save_dir_path"])
    return loader_loss


def get_params_lr_scheduler(configs):
    activate = bool(configs["lr_scheduler"]["activate"])
    scheduler_kwargs = configs["lr_scheduler"]["info"]
    scheduler_type = configs["lr_scheduler"]["scheduler_type"]
    return activate, scheduler_type, scheduler_kwargs


def calculate_parameters(model):
    qtd_model = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {qtd_model}")
    return


def run_training_experiment(model, train_loader, validation_loader, optimizer, custom_lr_scheduler,
                            criterion, configs
                            ):
    if configs['current_model']['save_model']:
        os.makedirs(configs['current_model']['model_dir'], exist_ok=True)

    calculate_parameters(model)

    for epoch in range(0, configs["epochs"]):
        train_loss = run_train_epoch(
            model, optimizer, criterion, train_loader,
            epoch
        )

        valid_loss = run_validation(
            model, criterion, validation_loader,
            epoch, configs
        )
        if custom_lr_scheduler is not None:
            if custom_lr_scheduler.scheduler_type == "reducelronplateau":
                custom_lr_scheduler.step(valid_loss)
            else:
                custom_lr_scheduler.step()
            print("Current learning rate:", custom_lr_scheduler.scheduler.get_last_lr()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    configs = read_yaml(args.config_file)

    print("============ Delete .wandb path ============")
    try:
        shutil.rmtree("wandb/")
    except:
        pass

    f_configurations = {}
    f_configurations = ToolsWandb.config_flatten(configs)

    model, train_loader, validation_loader, \
        optimizer, criterion = experiment_factory(configs)

    activate_lr_scheduler, scheduler_type, scheduler_kwargs = get_params_lr_scheduler(configs)

    if activate_lr_scheduler:
        custom_lr_scheduler = CustomLRScheduler(optimizer, scheduler_type, **scheduler_kwargs)
    else:
        custom_lr_scheduler = None

    if configs['reload_from_existing_model']['activate']:
        name_model = f"{configs['reload_from_existing_model']['model_dir']}/{configs['reload_from_existing_model']['model_name']}.pt"

        load_dict = torch.load(name_model)

        model.load_state_dict(load_dict['model_state_dict'])

    if configs["valid_on_the_fly"]["activate"]:
        valid_save_dir_path = configs["valid_on_the_fly"]["save_dir_path"]
        os.makedirs(valid_save_dir_path, exist_ok=True)
        clean_directory(valid_save_dir_path)

    if configs['wandb']["activate"]:
        wandb.init(project=configs['wandb']["project"],
                   reinit=True,
                   config=f_configurations,
                   entity=configs['wandb']["entity"],
                   save_code=False)

    run_training_experiment(
        model, train_loader, validation_loader, optimizer, custom_lr_scheduler,
        criterion, configs
    )

    torch.cuda.empty_cache()
    wandb.finish()
