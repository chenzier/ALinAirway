import os
import numpy as np


class DatasetInfo:
    def __init__(self, precrop_dataset):
        """
        初始化函数，接收预裁剪数据集的路径作为参数

        :param precrop_dataset: 预裁剪数据集的路径
        """
        self.precrop_dataset = precrop_dataset
        self.raw_path = precrop_dataset + "/image"
        self.label_path = precrop_dataset + "/label"
        self.raw_case_name_list = os.listdir(self.raw_path)
        self.label_case_name_list = os.listdir(self.label_path)

    def get_case_names(self, niigz_path, tag=None):
        """

        :return: 处理后的带有前缀且去重后的文件名数组
        """
        # 获取文件名并排序
        if tag is not None and tag.lower() == "exact09":
            niigz_image_path = niigz_path + "/train"
            niigz_label_path = niigz_path + "/train_label"
        else:
            niigz_image_path = niigz_path + "/image"
            niigz_label_path = niigz_path + "/label"
        names = os.listdir(niigz_image_path)
        names.sort()
        label_names = os.listdir(niigz_label_path)
        label_names.sort()

        # 为数据集生成带有特定前缀的文件名，并去除文件扩展名，通过 np.unique() 去重
        processed_names = []
        for name in names:
            if tag == "exact09" or "Exact09":
                processed_names.append("EXACT09_" + name.split(".")[0])
            processed_names.append(
                self.precrop_dataset.split("/")[-1] + "_" + name.split(".")[0]
            )
        processed_names = np.array(processed_names)
        processed_names = np.unique(processed_names)

        self.processed_names = processed_names

    def create_data_dict(self):
        """
        创建一个包含数据集图像和标签路径信息的数据字典

        :return: 包含数据集信息的数据字典
        """
        data_dict = dict()

        for _, name in enumerate(self.raw_case_name_list):

            data_dict[name.split(".")[0]] = {}
            data_dict[name.split(".")[0]]["image"] = self.raw_path + "/" + name
            data_dict[name.split(".")[0]]["label"] = self.label_path + "/" + name

        self.data_dict = data_dict


def split_train_test_sets(
    data_dict_EXACT09, data_dict_LIDC_IDRI, train_names, test_names
):
    """
    根据提供的名称将数据字典划分为训练集和测试集。

    参数:
    - data_dict_EXACT09 (dict): 包含 EXACT09 数据集的数据字典。
    - data_dict_LIDC_IDRI (dict): 包含 LIDC-IDRI 数据集的数据字典。
    - train_names (list): 包含训练集案例名称的列表。
    - test_names (list): 包含测试集案例名称的列表。

    返回:
    - dict: 包含训练集和测试集的字典。
    """
    train_test_set_dict = {"train": {}, "test": {}}

    # 处理 EXACT09 数据集
    for case in data_dict_EXACT09.keys():
        case_prefix = "_".join(case.split("_")[:2])
        if case_prefix in train_names:
            train_test_set_dict["train"][case] = data_dict_EXACT09[case]
        elif case_prefix in test_names:
            train_test_set_dict["test"][case] = data_dict_EXACT09[case]

    # 处理 LIDC-IDRI 数据集
    for case in data_dict_LIDC_IDRI.keys():
        case_prefix = "_".join(case.split("_")[:3])
        if case_prefix in train_names:
            train_test_set_dict["train"][case] = data_dict_LIDC_IDRI[case]
        elif case_prefix in test_names:
            train_test_set_dict["test"][case] = data_dict_LIDC_IDRI[case]

    train_test_set_dict["train_names"] = train_names
    train_test_set_dict["test_names"] = test_names

    return train_test_set_dict
