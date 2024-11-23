import numpy as np
import torch
import os
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import from_numpy as from_numpy
from matplotlib.colors import ListedColormap
import pickle
import sys
sys.path.append('../')  # 将上一层目录添加到模块搜索路径中
from func.model_arch2 import SegAirwayModel
from active_utils.random_crop import Random3DCrop_np, Normalization_np
from active_utils.embedding_tools import get_embeddings
from active_utils.dataset_process_tools import DatasetInfo

crop_size = (32, 128, 128)
windowMin = -1000
windowMax = 150
random3dcrop = Random3DCrop_np(crop_size)
normalization = Normalization_np(windowMin, windowMax)

LidcInfo = DatasetInfo("/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128")
LidcInfo.get_case_names("/mnt/wangc/LIDC", "lidc")

Exact09Info = DatasetInfo("/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128")
Exact09Info.get_case_names("/mnt/wangc/EXACT09/EXACT09_3D", "exact09")


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
model=SegAirwayModel(in_channels=1, out_channels=2)
model.to(device)
load_pkl = "/home/wangc/now/pure/checkpoint/abc_checkpoint_sample_org_33.pkl"
checkpoint = torch.load(load_pkl)
model.load_state_dict(checkpoint['model_state_dict'])
print(load_pkl)


N = len(LidcInfo.raw_case_name_list)
# N=10

embeddings_list, embeddings_dict = get_embeddings(
    LidcInfo.precrop_dataset,
    LidcInfo.raw_case_name_list,
    300000,
    model,
    device,
    random3dcrop,
    normalization,
    only_positive=True,
    need_embedding=1,
)
# 将列表中的NumPy数组堆叠成一个NumPy数组
stacked_embeddings_numpy = np.stack(embeddings_list, axis=0)
print(stacked_embeddings_numpy.shape)  # 输出应为 [N, 256, 4, 16, 16]

file_path = "/data/wangc/al_data/test1123/embedding/lidc_128_op_embeddings.pkl"
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'embeddings_list': embeddings_list, 'embeddings_dict': embeddings_dict}
    pickle.dump(data_to_save, file)
