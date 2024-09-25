import numpy as np
import torch
import os
import logging
import pickle
import sys
import argparse
import SimpleITK as sitk
from func.model_arch import SegAirwayModel
from func.model_run import semantic_segment_crop_and_cat
from func.post_process import post_process, add_broken_parts_to_the_result
from func.detect_tree import tree_detection
from func.ulti import get_df_of_line_of_centerline
from pipeline_use_func import (
    load_many_CT_img,
    get_metrics,
    get_the_skeleton_and_center_nearby_dict,
)

# 设置递归深度限制为10000，根据实际情况调整
sys.setrecursionlimit(10000)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Argument Parser for evaluation pipeline"
    )
    parser.add_argument(
        "--load_pkl", type=str, required=True, help="Path to the checkpoint .pth file"
    )
    parser.add_argument(
        "--metrics_save_path",
        type=str,
        required=True,
        help="Path to save the metrics .pkl file",
    )
    parser.add_argument(
        "--log_file", type=str, default="eval_pipeline.log", help="Path to the log file"
    )
    parser.add_argument(
        "--seg_result_path",
        type=str,
        default="",
        help="Path to save the segmentation results (optional)",
    )
    return parser.parse_args()


def eval_pipeline(load_pkl, metrics_save_path, test_names, seg_result_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegAirwayModel(in_channels=1, out_channels=2)
    model.to(device)
    checkpoint = torch.load(load_pkl)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = 0.5

    # 加载图像，需要更新一下数据集文件夹
    exact09_img_path = "/mnt/wangc/EXACT09/EXACT09_3D/train"
    lidc_img_path = "/mnt/wangc/LIDC/image"
    exact09_label_path = "/mnt/wangc/EXACT09/EXACT09_3D/train_label"
    lidc_label_path = "/mnt/wangc/LIDC/label"

    raw_img_dict = load_many_CT_img(
        exact09_img_path, lidc_img_path, test_names, dataset_type="image"
    )

    seg_processeds = []
    seg_processed_IIs = []
    logging.info(f"Start processing {len(raw_img_dict.keys())} cases")

    for i, (img_name, raw_img) in enumerate(raw_img_dict.items()):
        seg_result = semantic_segment_crop_and_cat(
            raw_img,
            model,
            device,
            crop_cube_size=[32, 128, 128],
            stride=[16, 64, 64],
            windowMin=-1000,
            windowMax=600,
        )
        logging.info(f"Case {i+1}: Segmentation done")

        seg_onehot = np.array(seg_result > threshold, dtype=np.int32)
        seg_onehot_comb = np.array((seg_onehot) > 0, dtype=np.int32)
        seg_processed, _ = post_process(seg_onehot_comb, threshold=threshold)

        (
            seg_slice_label_I,
            connection_dict_of_seg_I,
            number_of_branch_I,
            tree_length_I,
        ) = tree_detection(seg_processed, search_range=2)
        logging.info(f"Case {i+1}: Post-process done")

        seg_processed_II = add_broken_parts_to_the_result(
            connection_dict_of_seg_I,
            seg_result,
            seg_processed,
            threshold=threshold,
            search_range=10,
            delta_threshold=0.05,
            min_threshold=0.4,
        )

        (
            seg_slice_label_II,
            connection_dict_of_seg_II,
            number_of_branch_II,
            tree_length_II,
        ) = tree_detection(seg_processed_II, search_range=2)

        logging.info(f"Case {i+1}: Broken parts added and tree detection done")
        seg_processed_IIs.append(seg_processed)
        seg_processeds.append(seg_processed)

        if seg_result_path != "":
            sitk.WriteImage(
                sitk.GetImageFromArray(seg_processed),
                os.path.join(seg_result_path, f"{img_name}_segmentation.nii.gz"),
            )
            sitk.WriteImage(
                sitk.GetImageFromArray(seg_processed_II),
                os.path.join(
                    seg_result_path, f"{img_name}_segmentation_add_broken_parts.nii.gz"
                ),
            )

    logging.info("All cases processed, starting evaluation")

    # 导入test集的label
    label_dict = load_many_CT_img(
        exact09_label_path, lidc_label_path, test_names, dataset_type="label"
    )

    skeleton_dict = {}
    for i, label in label_dict.items():
        skeleton_dict[i] = get_the_skeleton_and_center_nearby_dict(
            label, search_range=2, need_skeletonize_3d=True
        )

    logging.info("Labels loaded")

    # 计算相应指标
    metrics_al = get_metrics(seg_processeds, label_dict, skeleton_dict)

    # 确保文件夹存在，如果不存在则创建它
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    # 保存到文件
    with open(metrics_save_path, "wb") as file:
        data_to_save = {"metrics_al": metrics_al}
        pickle.dump(data_to_save, file)

    logging.info("Evaluation done and metrics saved")


if __name__ == "__main__":
    args = parse_arguments()
    setup_logging(args.log_file)

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
    logging.info(
        f"Arguments: load_pkl={args.load_pkl}, metrics_save_path={args.metrics_save_path}, "
        f"seg_result_path={args.seg_result_path}, test_names={test_names}"
    )
    eval_pipeline(
        args.load_pkl, args.metrics_save_path, test_names, args.seg_result_path
    )
