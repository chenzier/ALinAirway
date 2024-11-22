import numpy as np
import torch
import os
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import from_numpy as from_numpy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


import pickle
import sys

sys.path.append("../")  # 将上一层目录添加到模块搜索路径中

from active_utils.dataset_process_tools import DatasetInfo
from active_utils.embedding_tools import get_embeddings, load_partial_embeddings
from active_utils.visualize_tools import (
    visualize_and_return_indices,
    show_all_2d_img_with_labels,
)
from active_utils.cluster_tools import kmeans
from active_utils.file_tools import load_obj, save_obj

batch_size = 50
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
test_names = [
    "LIDC_IDRI_0066",
    "LIDC_IDRI_0328",
    "LIDC_IDRI_0376",
    "LIDC_IDRI_0441",
    "LIDC_IDRI_0744",
    "LIDC_IDRI_1004",
    "EXACT09_CASE13",
    "EXACT09_CASE08",
    "EXACT09_CASE01",
    "EXACT09_CASE05",
]

LidcInfo = DatasetInfo("/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128")
LidcInfo.get_case_names("/mnt/wangc/LIDC", "lidc")

Exact09Info = DatasetInfo("/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128")
Exact09Info.get_case_names("/mnt/wangc/EXACT09/EXACT09_3D", "exact09")

crop_size = ["128", "256"]
file_insert = crop_size[0]
exact_embedding_path = (
    f"/home/wangc/now/NaviAirway/saved_var/exact09_{file_insert}_op_embeddings_data.pkl"
)
lidc_embedding_path = (
    f"/home/wangc/now/NaviAirway/saved_var/lidc_{file_insert}_op_embeddings_data.pkl"
)

exact_lidc_concatenated_array, merged_dict, merged_list = load_partial_embeddings(
    exact_embedding_path, lidc_embedding_path, train_names=None, test_names=test_names
)

data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)
# 需要把数据放到GPU上
X_t = from_numpy(X_t).float().to(device2)
X_t_expanded = X_t.unsqueeze(1)
X_t_expanded = X_t_expanded.to(device2)
N = X_t.shape[0]

num_cluster = 2
cluster_labels, cluster_centers = kmeans(
    X=X_t, num_clusters=num_cluster, init=None, distance="euclidean", device=device2
)
cluster_centers_expanded = cluster_centers.unsqueeze(0)
cluster_centers_expanded = cluster_centers_expanded.to(device2)


uncertainy_dict = {}
for i in range(0, N, batch_size):
    # Select a batch of data
    X_batch = X_t_expanded[i : i + batch_size]

    # Calculate distances for the batch
    distances_batch = torch.sqrt(
        torch.sum((X_batch - cluster_centers_expanded) ** 2, dim=2)
    )

    # Calculate uncertainty for the batch
    uncertainy_batch = torch.abs(distances_batch[:, 0] - distances_batch[:, 1])

    # Update uncertainy_dict with batch results
    for j in range(batch_size):
        index = i + j
        if index < N:
            uncertainy_dict[merged_list[index]] = uncertainy_batch[j].cpu().numpy()


res_path = "/home/wangc/now/NaviAirway/saved_var/ae1_uncertainy_128_data.pkl"
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(res_path), exist_ok=True)

# 保存到文件
with open(file_path, "wb") as file:
    data_to_save = {"uncertainy_dict": uncertainy_dict}
    pickle.dump(data_to_save, file)
