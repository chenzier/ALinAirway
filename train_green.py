import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
import gc
import sys
import logging

sys.path.append("..")
sys.path.append("/home/wangc/now/pure/ALinAirway/func")  # 根据实际情况调整路径
from func.load_dataset import airway_dataset
from func.model_arch import SegAirwayModel
from func.loss_func import (
    dice_loss_weights,
    dice_accuracy,
    dice_loss_power_weights,
)
from func.ulti import load_obj
import argparse

# python train_with_args.py --use_gpu cuda:7 --checkpoint_path /home/wangc/now/NaviAirway/checkpoint/check_point_0119/ae120_test_checkpoint.pth --data_info_path /home/wangc/now/NaviAirway/saved_objs/for_128_objs/training_info_0119/ae1_info_20.pkl --log_name training_test_0811.log
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

    return parser.parse_args()


args = parse_arguments()

# Configure logging

ck_dir = "/home/wangc/now/pure/save_checkpoint"  # 模型存储路径
log_dir = "/home/wangc/now/pure/save_log"  # 日志存储路径

checkpoint_path = os.path.join(ck_dir, args.save_name + ".pth")
log_name = os.path.join(log_dir, args.save_name + ".log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_name), logging.StreamHandler(sys.stdout)],
)

# Log all arguments
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
batch_size = 4
train_file_format = ".nii.gz"
crop_size = (32, 128, 128)
windowMin_CT_img_HU = -1000
windowMax_CT_img_HU = 600
model_save_freq = 1
num_workers = 4

# Init model
model = SegAirwayModel(in_channels=1, out_channels=2)
device = torch.device(use_gpu if torch.cuda.is_available() else "cpu")

logging.info(f"Device: {device}")
model.to(device)

# Load checkpoint if necessary
if need_resume and os.path.exists(checkpoint_path):
    logging.info(f"Resuming model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load dataset
dataset_info_org = load_obj(data_info_path)
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
start_time = time.time()

for ith_epoch in range(max_epoch):

    sampler_of_airways_org = RandomSampler(
        train_dataset_org,
        num_samples=min(num_samples_of_each_epoch, len(dataset_info_org)),
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

        if ith_batch in [1, 100, len_dataset_loader]:
            # if ith_batch != -1:
            logging.info(
                f"epoch [{ith_epoch + 1}/{max_epoch}]\t"
                f"batch [{ith_batch}/{len_dataset_loader}]\t"
                f"time(sec) {time_consumption:.2f}\t"
                f"loss {loss.item():.4f}\t"
                f"acc {accuracy.item() * 100:.2f}%\t"
                f"fore pix {fore_pix_per * 100:.2f}%\t"
                f"back pix {back_pix_per * 100:.2f}%\t"
            )

    del dataset_loader
    gc.collect()

    if (ith_epoch + 1) % model_save_freq == 0:
        logging.info(f"Epoch {ith_epoch + 1}: Saving model")
        model.to(torch.device("cpu"))
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        model.to(device)

logging.info(f"Model saved at {checkpoint_path}")
logging.info("Training completed.")
