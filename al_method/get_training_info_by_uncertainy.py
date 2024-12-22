# 执行这个文件必须在python 3.8版本下
import os
import numpy as np
import pickle
from active_utils.file_tools import save_obj, load_obj
from active_utils.dataset_process_tools import DatasetInfo, split_train_test_sets
from active_utils.select_tools import select_from_uncertainy
import random

LidcInfo = DatasetInfo("/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128", "lidc", 128)
LidcInfo.get_case_names("/mnt/wangc/LIDC", "lidc")

Exact09Info = DatasetInfo(
    "/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128", "exact09", 128
)
Exact09Info.get_case_names("/mnt/wangc/EXACT09/EXACT09_3D", "exact09")

LidcInfo.create_data_dict()
Exact09Info.create_data_dict()

names = np.concatenate((Exact09Info.processed_names, LidcInfo.processed_names))

test_names = [
    "LIDC_IDRI_0066",
    "LIDC_IDRI_0328",
    "LIDC_IDRI_0376",
    "LIDC_IDRI_0441",
    "EXACT09_CASE13",
    "LIDC_IDRI_0744",
    "EXACT09_CASE08",
    "EXACT09_CASE01",
    "EXACT09_CASE05",
    "LIDC_IDRI_1004",
]

train_names = []
for name in names:
    if name not in test_names:
        train_names.append(name)
train_names = np.array(train_names)

train_test_set_dict_128 = split_train_test_sets(
    Exact09Info.data_dict, LidcInfo.data_dict, train_names, test_names
)

data_dict_org = load_obj(
    "/home/wangc/now/pure/saved_objs/for_128_objs/data_dict_org.pkl"
)
data_dict_only_negtive = load_obj(
    "/home/wangc/now/pure/saved_objs/for_128_objs/data_dict_only_negtive"
)
num = 70
new_num = 0.01 * num
uncertainy_path = (
    "/data/wangc/al_data/test1123/uncertainy/uncertainy_entropy_1221_1838.pkl"
)

save_path = f"/home/wangc/now/pure/saved_objs/for_128_objs/training_info_1218/num{num}_{os.path.basename(uncertainy_path)}"
num1, num2, num3 = select_from_uncertainy(
    uncertainy_path, data_dict_org, data_dict_only_negtive, new_num, save_path=save_path
)
print(num1, num2, num3)
print(new_num)
