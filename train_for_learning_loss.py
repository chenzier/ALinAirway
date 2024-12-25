import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
import gc
import sys
import argparse
import numpy as np
import pickle
import SimpleITK as sitk
import yaml
import importlib
import logging
from func.load_dataset import airway_dataset, airway_dataset2
from func.model_arch_for_learning_loss import SegAirwayModel, LossNet
from func.loss_func import (
    dice_loss_weights,
    dice_accuracy,
    dice_loss_power_weights,
    dice_loss_power_weights_for_learningloss,
    dice_loss_weights_for_learningloss,
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

    return parser.parse_args()


def train_model_for_learning_loss(
    models,
    optimizers,
    train_dataset_org,
    device,
    max_epoch,
    num_samples_of_each_epoch,
    batch_size,
    num_workers,
    model_save_freq,
    checkpoint_path,
):
    MARGIN = 1.0  # xi
    WEIGHT = 1.0  # lambda
    start_time = time.time()

    # Loss Prediction Loss
    def LossPredLoss(input, target, margin=1.0, reduction="mean"):
        if len(input) % 2 != 0:
            input = input[:-1]
            target = target[:-1]
        assert len(input) % 2 == 0, "the batch size is not even."
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[
            : len(input) // 2
        ]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[: len(target) // 2]
        target = target.detach()

        one = (
            2 * torch.sign(torch.clamp(target, min=0)) - 1
        )  # 1 operation which is defined by the authors

        if reduction == "mean":
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0)  # Note that the size of input is already halved
        elif reduction == "none":
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss

    seg_model = models["backbone"]
    loss_model = models["module"]
    seg_model.train()
    loss_model.train()

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

            img_output = seg_model(img_input)
            features = seg_model.get_embedding(img_input)
            # print(len(features), type(features[0]))
            # for i in range(len(features)):
            #     print(i, features[i].shape)

            target_loss = dice_loss_weights_for_learningloss(
                img_output[:, 0, :, :, :], groundtruth_background, weights
            ) + dice_loss_power_weights_for_learningloss(
                img_output[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2
            )
            pred_loss = loss_model(features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_backbone_loss = torch.sum(target_loss)
            print(
                f"batch [{ith_batch}/{len(dataset_loader)}]",
                pred_loss.shape,
                target_loss.shape,
            )
            m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss = m_backbone_loss + WEIGHT * m_module_loss

            optimizers["backbone"].zero_grad()
            optimizers["module"].zero_grad()
            loss.backward()
            optimizers["backbone"].step()
            optimizers["module"].step()
            time_consumption = time.time() - start_time

            if ith_batch % 100 == 0:
                logging.info(
                    f"epoch [{ith_epoch + 1}/{max_epoch}]\t"
                    f"batch [{ith_batch}/{len(dataset_loader)}]\t"
                    f"time(sec) {time_consumption:.2f}\t"
                    f"loss {loss.item():.4f}\t"
                    f"fore pix {fore_pix_per * 100:.2f}%\t"
                    f"back pix {back_pix_per * 100:.2f}%\t"
                )
        del dataset_loader
        gc.collect()

        if (ith_epoch + 1) % model_save_freq == 0:
            logging.info(f"Epoch {ith_epoch + 1}: Saving model")
            seg_model.to(torch.device("cpu"))
            loss_model.to(torch.device("cpu"))
            torch.save(
                {
                    "seg_model_state_dict": seg_model.state_dict(),
                    "loss_model_state_dict": loss_model.state_dict(),
                },
                checkpoint_path,
            )
            seg_model.to(device)
            loss_model.to(device)


def eval_for_learning_loss(
    models,
    checkpoint_path,
    unlabeled_loader,
    device,
    batch_size,
    num_workers,
    output_dir,  # 添加保存路径参数
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建存储不确定性的字典，键为样本名称，值为pred_loss
    uncertainty = {}

    # 加载模型检查点
    checkpoints = torch.load(checkpoint_path)
    models["backbone"].load_state_dict(checkpoints["seg_model_state_dict"])
    models["module"].load_state_dict(checkpoints["loss_model_state_dict"])

    # 将模型移动到对应的设备（GPU 或 CPU）
    models["backbone"].to(device)
    models["module"].to(device)

    # 创建 unlabelled dataset 的采样器
    sampler_of_unlabeled_dataset = RandomSampler(
        unlabeled_loader,
        num_samples=min(len(unlabeled_loader), len(unlabeled_loader)),
        replacement=True,
    )

    # 创建 DataLoader
    unlabeled_loader = DataLoader(
        unlabeled_loader,
        batch_size=batch_size,
        sampler=sampler_of_unlabeled_dataset,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 1),
    )
    names_list = unlabeled_dataset.get_name_list()
    # 在不更新梯度的情况下进行预测
    with torch.no_grad():
        for batch in unlabeled_loader:
            img_input = batch["image"].float().to(device)
            idxs = batch["idx"]
            print(idxs)
            # 前向传播获取模型特征
            features = models["backbone"].get_embedding(img_input)
            pred_loss = models["module"](features)  # 获取模型预测的loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            # 将pred_loss与样本名称保存到字典中
            for idx, loss in zip(idxs, pred_loss):
                name = names_list[idx]
                uncertainty[name] = loss.item()  # 将预测loss转为标量并保存

    # 将字典保存到文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存 uncertainty 字典为 pickle 文件
    output_path = os.path.join(output_dir, "uncertainty.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(uncertainty, f)


if __name__ == "__main__":

    # 导入命令行参数
    args = parse_arguments()

    # 导入config参数
    config = load_config("../config.yaml")
    exact09_img_path = config["exact09"]["img_path"]
    lidc_img_path = config["lidc"]["img_path"]
    exact09_label_path = config["exact09"]["label_path"]
    lidc_label_path = config["lidc"]["label_path"]

    ck_dir = config["eval_use"]["ck_dir"]  # 模型存储路径
    checkpoint_path = os.path.join(ck_dir, args.save_name + ".pth")

    use_gpu = str(args.use_gpu)
    checkpoint_path = str(checkpoint_path)
    data_info_path = str(args.data_info_path)

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

    log_dir = config["eval_use"]["log_dir"]  # 日志存储路径
    log_name = os.path.join(log_dir, args.save_name + ".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_name), logging.StreamHandler(sys.stdout)],
    )

    logging.info("Script Arguments:")

    # Init model
    seg_model = SegAirwayModel(in_channels=1, out_channels=2)
    loss_model = LossNet()
    models = {"backbone": seg_model, "module": loss_model}
    device = torch.device(use_gpu if torch.cuda.is_available() else "cpu")
    models["backbone"].to(device)
    models["module"].to(device)

    batch_size = 8

    # Optimizer todo...

    seg_optimizer = torch.optim.Adam(models["backbone"].parameters(), lr=learning_rate)
    loss_optimizer = torch.optim.Adam(models["module"].parameters(), lr=learning_rate)
    optimizers = {"backbone": seg_optimizer, "module": loss_optimizer}

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

    torch.set_num_threads(2)
    # train_model_for_learning_loss(
    #     models,
    #     optimizers,
    #     train_dataset_org,
    #     device,
    #     max_epoch,
    #     num_samples_of_each_epoch,
    #     batch_size,
    #     num_workers,
    #     model_save_freq,
    #     checkpoint_path,
    # )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    unlabeled_info_path = "../saved_objs/for_128_objs/data_dict_org.pkl"
    unlabeled_dataset_info = load_obj(unlabeled_info_path)
    output_dir = (
        "/data/wangc/al_data/test1123/uncertainy/uncertainy_learning_loss_1223_1655.pkl"
    )
    unlabeled_dataset = airway_dataset2(unlabeled_dataset_info)
    unlabeled_dataset.set_para(
        file_format=train_file_format,
        crop_size=crop_size,
        windowMin=windowMin_CT_img_HU,
        windowMax=windowMax_CT_img_HU,
        need_tensor_output=True,
        need_transform=True,
    )
    eval_for_learning_loss(
        models,
        checkpoint_path,
        unlabeled_dataset,
        device,
        batch_size,
        num_workers,
        output_dir,
    )
