import numpy as np
import os
from func.ulti import (
    load_one_CT_img,
)

from skimage.morphology import skeletonize_3d


def load_many_CT_img(exact09_img_path, lidc_img_path, filenames, dataset_type="image"):

    img_dict = {}
    if dataset_type == "image":
        for filename in filenames:
            if filename[:3] == "LID":  # 这个if的设计是根据test_names来的
                img_dict[filename] = load_one_CT_img(
                    os.path.join(lidc_img_path, filename)
                )
            if filename[:3] == "EXA":
                img_dict[filename] = load_one_CT_img(
                    os.path.join(exact09_img_path, filename[8:])
                )
    if dataset_type == "label":
        for filename in filenames:
            # print(filename)
            if filename[:3] == "LID":  # 这个if的设计是根据test_names来的
                file_reset_name = filename[:14] + "_label.nii.gz"
                # print(file_reset_name)
                img_dict[filename] = load_one_CT_img(
                    os.path.join(lidc_img_path, file_reset_name)
                )
            if filename[:3] == "EXA":
                file_reset_name = filename[8:14] + "_label.nii.gz"
                # print(file_reset_name)
                img_dict[filename] = load_one_CT_img(
                    os.path.join(exact09_img_path, file_reset_name)
                )

    return img_dict


def get_metrics(seg_processed_IIs, label_dict, skeleton_dict):

    def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
        label_branch = label_skeleton * label_parsing
        label_branch_flat = label_branch.flatten()
        label_branch_bincount = np.bincount(label_branch_flat)[1:]
        total_branch_num = label_branch_bincount.shape[0]
        pred_branch = label_branch * pred
        pred_branch_flat = pred_branch.flatten()
        pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
        if total_branch_num != pred_branch_bincount.shape[0]:
            lack_num = total_branch_num - pred_branch_bincount.shape[0]
            pred_branch_bincount = np.concatenate(
                (pred_branch_bincount, np.zeros(lack_num))
            )
        branch_ratio_array = pred_branch_bincount / label_branch_bincount
        branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
        detected_branch_num = np.count_nonzero(branch_ratio_array)
        detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
        return total_branch_num, detected_branch_num, detected_branch_ratio

    def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
        pred = pred.flatten()
        label = label.flatten()
        intersection = np.sum(pred * label)
        dice_coefficient_score = round(
            ((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth))
            * 100,
            2,
        )
        return dice_coefficient_score

    def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
        pred = pred.flatten()
        label_skeleton = label_skeleton.flatten()
        tree_length = round(
            (np.sum(pred * label_skeleton) + smooth)
            / (np.sum(label_skeleton) + smooth)
            * 100,
            2,
        )
        return tree_length

    def false_positive_rate_calculation(pred, label, smooth=1e-5):
        pred = pred.flatten()
        label = label.flatten()
        fp = np.sum(pred - pred * label) + smooth
        fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
        return fpr

    def false_negative_rate_calculation(pred, label, smooth=1e-5):
        pred = pred.flatten()
        label = label.flatten()
        fn = np.sum(label - pred * label) + smooth
        fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
        return fnr

    def iou_calculation(pred, label, smooth=1e-5):
        # Flatten the predictions and labels
        pred = pred.flatten()
        label = label.flatten()

        # Calculate TP, FP, and FN
        tp = np.sum(pred * label)
        fp = np.sum(pred - pred * label)
        fn = np.sum(label - pred * label)

        # Calculate IoU
        iou = tp / (tp + fp + fn + smooth)

        # Round the IoU to 3 decimal places

        return iou

    def sensitivity_calculation(pred, label):
        sensitivity = round(100 - false_negative_rate_calculation(pred, label), 3)
        return sensitivity

    def specificity_calculation(pred, label):
        specificity = round(100 - false_positive_rate_calculation(pred, label), 3)
        return specificity

    def precision_calculation(pred, label, smooth=1e-5):
        pred = pred.flatten()
        label = label.flatten()
        tp = np.sum(pred * label) + smooth
        precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
        return precision

    def DSC(sensitivity, precision):
        s, p = sensitivity, precision
        return (2 * p * s) / (p + s)

    i = 0
    many_metrics = {}
    for key in label_dict.keys():
        print(key, i)
        metrics = {}
        pred = seg_processed_IIs[i]
        label = label_dict[key]
        label_skeleton = skeleton_dict[key]
        i += 1
        total_branch_num, detected_branch_num, detected_branch_ratio = (
            branch_detected_calculation(pred, label, label_skeleton)
        )
        dice_coefficient_score = dice_coefficient_score_calculation(
            pred, label, smooth=1e-5
        )
        tree_length = tree_length_calculation(pred, label_skeleton, smooth=1e-5)
        fpr = false_positive_rate_calculation(pred, label, smooth=1e-5)
        fnr = false_negative_rate_calculation(pred, label, smooth=1e-5)
        sensitivity = sensitivity_calculation(pred, label)
        specificity = specificity_calculation(pred, label)
        precision = precision_calculation(pred, label, smooth=1e-5)
        dsc = DSC(sensitivity, precision)
        iou = iou_calculation(pred, label, smooth=1e-5)

        metrics["detected_branch_num"] = detected_branch_num
        metrics["detected_branch_ratio"] = detected_branch_ratio
        metrics["tree_length"] = tree_length
        metrics["fpr"] = fpr
        metrics["fnr"] = fnr
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = total_branch_num
        metrics["precision"] = precision
        metrics["DSC"] = dsc
        metrics["iou"] = iou

        many_metrics[key] = metrics
    #     print(metrics)
    # print(many_metrics)
    return many_metrics


def get_the_skeleton_and_center_nearby_dict(
    seg_input, search_range=10, need_skeletonize_3d=True
):
    # 传入参数need_skeletonize_3d=True
    # 使用3D骨架化函数骨架化分割后的输入图像，并将骨架化图像转换为中心图
    if need_skeletonize_3d:
        center_map = np.array(skeletonize_3d(seg_input) > 0, dtype=np.int32)
    else:
        center_map = seg_input

    center_dict = {}
    nearby_dict = {}
    # 对于中心图中的每个中心，获取中心的坐标以及给定搜索范围内附近中心的坐标。
    # print(center_map.shape,seg_input.shape)#(224, 512, 512) (224, 512, 512)
    # 骨架化得到的中心图center_map和分割图像seg_input都是二值化的三维张量，其元素值为0或1，其中1表示该像素属于支气管的中心线

    center_locs = np.where(center_map > 0)
    # np.where(center_map>0)返回一个元组，元组中包含了三个数组，
    # 分别代表了中心图center_map中值大于0的像素的三维坐标。
    # 具体来说，假设中心图的形状为(D, H, W)，
    # 则center_locs[0]表示中心图中所有值大于0的像素在第一个维度（深度方向）上的坐标，
    # 即一个长度为n的一维数组；
    # center_locs[1]表示所有像素在第二个维度（高度方向）上的坐标，
    # 也是一个长度为n的一维数组；
    # center_locs[2]表示所有像素在第三个维度（宽度方向）上的坐标，
    # 同样也是一个长度为n的一维数组。
    # 其中，n是中心图中值大于0的像素数量，即支气管的中心线像素数量

    # 这部分代码是为每个中心点在字典center_dict中分配一个唯一的标识符，并将该标识符添加到中心图center_map中。
    base_count = 1
    for i in range(len(center_locs[0])):
        # 将该中心点在字典center_dict中分配一个唯一的标识符i+base_count，
        # 该标识符由当前循环次数i和base_count相加得到。
        # 然后，将该中心点的坐标(center_locs[0][i], center_locs[1][i], center_locs[2][i])作为值，将标识符作为键，添加到字典center_dict中。
        center_dict[i + base_count] = [
            center_locs[0][i],
            center_locs[1][i],
            center_locs[2][i],
        ]  # center_locs[0]、[1]、[2]是中心图的三维坐标
        # 将中心图center_map中该中心点的像素值赋为其唯一标识符，即i+base_count。
        # 这里的目的是方便后面的操作中使用字典center_dict来查找每个中心点的坐标
        center_map[center_locs[0][i], center_locs[1][i], center_locs[2][i]] = (
            i + base_count
        )

    return center_map


from datetime import datetime


def print_message_with_timestamp(message):
    # 获取当前时间戳
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 打印消息和时间戳
    print(f"{message} {current_time}")
