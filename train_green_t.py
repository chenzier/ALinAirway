import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
import gc
import sys
import logging
import argparse
import numpy as np
import pickle
import SimpleITK as sitk
import yaml
import wandb
import importlib

from func.load_dataset import airway_dataset
from func.loss_func import (
    dice_loss_weights,
    dice_accuracy,
    dice_loss_power_weights,
)
from func.model_run import semantic_segment_crop_and_cat
from func.post_process import post_process, add_broken_parts_to_the_result
from func.detect_tree import tree_detection
from func.ulti import get_df_of_line_of_centerline, load_obj
from func.eval_use_func import (
    load_many_CT_img,
    get_metrics,
    get_the_skeleton_and_center_nearby_dict,
)

sys.setrecursionlimit(100000)


def update_dataset_paths(dataset, old_prefix, new_prefix):
    for key, value in dataset.items():
        # 去除指定前缀，并添加新的前缀
        value["image"] = new_prefix + value["image"].replace(old_prefix, "")
        value["label"] = new_prefix + value["label"].replace(old_prefix, "")
    return dataset


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for your script")

    parser.add_argument(
        "--use_gpu",
        type=str,
        default="cuda:0",
        help='Specify GPU usage (e.g., "--use_gpu cuda:0")',
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help='Specify a base name for saving the checkpoint and log (e.g., "random20_dsc_1214_1513")',
    )
    parser.add_argument(
        "--data_info_path",
        type=str,
        default="",
        help='Specify dataset info load address (e.g., "--data_info_path /path/to/load")',
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model architecture module name"
    )
    parser.add_argument(
        "--if_use_wandb",
        type=bool,
        default=False,  # 设置默认值为False，即默认不使用wandb
        help='Specify whether to use wandb (e.g., "--if_use_wandb True")',
    )

    return parser.parse_args()
def train_model(
    model,
    optimizer,
    train_dataset_org,
    device,
    max_epoch,
    num_samples_of_each_epoch,
    batch_size,
    num_workers,
    model_save_freq,
    checkpoint_path,
):
    """
    训练模型的函数
    :param model: 待训练的模型实例
    :param optimizer: 优化器实例
    :param train_dataset_org: 训练数据集实例
    :param device: 训练使用的设备（GPU或CPU）
    :param max_epoch: 最大训练轮数
    :param num_samples_of_each_epoch: 每轮采样的数据数量
    :param batch_size: 批次大小
    :param num_workers: 数据加载的工作线程数量
    :param model_save_freq: 模型保存的频率（多少轮保存一次）
    :param checkpoint_path: 模型检查点保存路径
    """
    start_time = time.time()
    for ith_epoch in range(max_epoch):
        sampler_of_airways_org = RandomSampler(
            train_dataset_org,
            num_samples=min(num_samples_of_each_epoch, len(train_dataset_org)),
            replacement=True,
        )
        dataset_loader = DataLoader(
            train_dataset_org,
            batch_size=batch_size,
            sampler=sampler_of_airways_org,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 1),
        )

        len_dataset_loader = len(dataset_loader)
        for ith_batch, batch in enumerate(dataset_loader):
            img_input = batch["image"].float().to(device)
            groundtruth_foreground = batch["label"].float().to(device)
            groundtruth_background = 1 - groundtruth_foreground

            fore_pix_num = torch.sum(groundtruth_foreground)
            back_pix_num = torch.sum(groundtruth_background)
            fore_pix_per = fore_pix_num / (fore_pix_num + back_pix_num)
            back_pix_per = back_pix_num / (fore_pix_num + back_pix_num)

            weights = (
                torch.exp(back_pix_per)
                / (torch.exp(fore_pix_per) + torch.exp(back_pix_per))
                * torch.eq(groundtruth_foreground, 1).float()
                + torch.exp(fore_pix_per)
                / (torch.exp(fore_pix_per) + torch.exp(back_pix_per))
                * torch.eq(groundtruth_foreground, 0).float()
            ).to(device)
            img_output = model(img_input)

            loss = dice_loss_weights(
                img_output[:, 0, :, :, :], groundtruth_background, weights
            ) + dice_loss_power_weights(
                img_output[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2
            )
            accuracy = dice_accuracy(img_output[:, 1, :, :, :], groundtruth_foreground)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_consumption = time.time() - start_time

            if ith_batch % 100 == 0:
                logging.info(
                    f"epoch [{ith_epoch + 1}/{max_epoch}]\t"
                    f"batch [{ith_batch}/{len_dataset_loader}]\t"
                    f"time(sec) {time_consumption:.2f}\t"
                    f"loss {loss.item():.4f}\t"
                    f"acc {accuracy.item() * 100:.2f}%\t"
                    f"fore pix {fore_pix_per * 100:.2f}%\t"
                    f"back pix {back_pix_per * 100:.2f}%\t"
                )

            if args.if_use_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "accuracy": accuracy.item(),
                    }
                )

        del dataset_loader
        gc.collect()

        if (ith_epoch + 1) % model_save_freq == 0:
            logging.info(f"Epoch {ith_epoch + 1}: Saving model for {args.save_name}")
            model.to(torch.device("cpu"))
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
            model.to(device)

    logging.info(f"Model saved at {checkpoint_path}")
    logging.info("Training completed.")


if __name__ == "__main__":

    # 导入命令行参数
    seed_value = 42
    torch.manual_seed(seed_value)
    args = parse_arguments()

    # 导入本次运行模型
    try:
        model_module = importlib.import_module(f"func.model_arch_{args.model}")
        SegAirwayModel = getattr(model_module, "SegAirwayModel")
    except ModuleNotFoundError:
        print(f"Error: Model architecture func.model_arch_{args.model} not found.")
        exit(1)

    # 导入config参数
    config = load_config("../config.yaml")
    exact09_img_path = config["exact09"]["img_path"]
    lidc_img_path = config["lidc"]["img_path"]
    exact09_label_path = config["exact09"]["label_path"]
    lidc_label_path = config["lidc"]["label_path"]

    ck_dir = config["eval_use"]["ck_dir"]  # 模型存储路径
    log_dir = config["eval_use"]["log_dir"]  # 日志存储路径
    metrics_folder_path = config["eval_use"][
        "metrics_folder_path"
    ]  # 评估时使用的metrics路径

    checkpoint_path = os.path.join(ck_dir, args.save_name + ".pth")
    log_name = os.path.join(log_dir, args.save_name + ".log")

    os.makedirs(
        metrics_folder_path, exist_ok=True
    )  # 根据save_name生成metrics_save_path完整路径，后缀为.pkl
    metrics_save_path = os.path.join(metrics_folder_path, f"{args.save_name}.pkl")

    # 新建info日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_name), logging.StreamHandler(sys.stdout)],
    )

    logging.info("Script Arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    use_gpu = str(args.use_gpu)
    checkpoint_path = str(checkpoint_path)
    data_info_path = str(args.data_info_path)

    if checkpoint_path == "" or data_info_path == "":
        logging.error(
            "Checkpoint save address or dataset info load address cannot be empty."
        )
        assert False

    # Configuration
    need_resume = True
    learning_rate = 1e-5
    max_epoch = 50
    freq_switch_of_train_mode_high_low_generation = 1
    num_samples_of_each_epoch = 20000
    train_file_format = ".nii.gz"
    crop_size = (32, 128, 128)
    windowMin_CT_img_HU = -1000
    windowMax_CT_img_HU = 600
    model_save_freq = 1
    num_workers = 4

    # Init model
    model = SegAirwayModel(in_channels=1, out_channels=2)
    device = torch.device(use_gpu if torch.cuda.is_available() else "cpu")
    model_message, flag = model.model_info()
    print(flag)
    if flag in config["batch_size_list"].keys():
        batch_size = config["batch_size_list"][flag]
    else:
        batch_size = 8
    logging.info(f"Batch size: {batch_size}")
    logging.info(model_message)
    logging.info(f"Device: {device}")
    model.to(device)
    if args.if_use_wandb:
        wandb.init(
            project=args.model,
            config={
                "save_name": args.save_name,
                "learning_rate": learning_rate,
                "max_epoch": max_epoch,
                "batch_size": batch_size,
                # 可以继续添加其他想要记录的超参数
            },
        )
    # Load checkpoint if necessary
    if need_resume and os.path.exists(checkpoint_path):
        logging.info(f"Resuming model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load dataset
    dataset_info_org = load_obj(data_info_path)
    if config["is_change_prefix"]["is_change"]:
        old_prefix = config["is_change_prefix"]["old_prefix"]
        new_prefix = config["is_change_prefix"]["new_prefix"]
        dataset_info_org = update_dataset_paths(
            dataset_info_org, old_prefix, new_prefix
        )

    train_dataset_org = airway_dataset(dataset_info_org)
    train_dataset_org.set_para(
        file_format=train_file_format,
        crop_size=crop_size,
        windowMin=windowMin_CT_img_HU,
        windowMax=windowMax_CT_img_HU,
        need_tensor_output=True,
        need_transform=True,
    )

    logging.info(f"Total epochs: {max_epoch}")
    logging.info(f"Length of dataset: {len(dataset_info_org)}")

    torch.set_num_threads(2)
    train_model(
        model,
        optimizer,
        train_dataset_org,
        device,
        max_epoch,
        num_samples_of_each_epoch,
        batch_size,
        num_workers,
        model_save_freq,
        checkpoint_path,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_names = [
        "LIDC_IDRI_0066.nii.gz",
        "LIDC_IDRI_0328.nii.gz",
        "LIDC_IDRI_0376.nii.gz",
        "LIDC_IDRI_0441.nii.gz",
        "EXACT09_CASE13.nii.gz",
        "LIDC_IDRI_0744.nii.gz",
        "EXACT09_CASE08.nii.gz",
        "EXACT09_CASE01.nii.gz",
        "EXACT09_CASE05.nii.gz",
        "LIDC_IDRI_1004.nii.gz",
    ]

    eval_pipeline(
        model,
        checkpoint_path,
        test_names,
        "",
        device,
        metrics_save_path,
        exact09_img_path,
        lidc_img_path,
        exact09_label_path,
        lidc_label_path,
    )
    if args.if_use_wandb:
        wandb.finish()
