import numpy as np
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import from_numpy as from_numpy
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import sys
sys.path.append('../')  # 将上一层目录添加到模块搜索路径中
from al_method.active_learning_utils import process_images,visualize_and_return_indices,show_all_2d_img_with_labels,kmeans,save_obj,load_obj,load_partial_embeddings
import pickle
from func.model_arch2 import SegAirwayModel

import torch.utils.data as data_utils


import edt


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# 设置分布式环境
# local_rank = int(os.environ.get("LOCAL_RANK", 0))



crop_size=['128','256']
file_insert=crop_size[0]
Precrop_dataset_for_train_path = f"/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_{file_insert}"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
# print(len(raw_case_name_list))

lidc_dataset_for_train_path=f'/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_{file_insert}'
lidc_dataset_for_train_raw_path =lidc_dataset_for_train_path+"/image"
lidc_dataset_for_train_label_path = lidc_dataset_for_train_path+"/label"
lidc_raw_case_name_list = os.listdir(lidc_dataset_for_train_raw_path)
# print(len(lidc_raw_case_name_list))

# 从文件加载
# file_path1='/home/wangc/now/NaviAirway/saved_var/exact09_256_embeddings_data.pkl'
# file_path2='/home/wangc/now/NaviAirway/saved_var/lidc_256_embeddings_data.pkl'

file_path1=f'/home/wangc/now/NaviAirway/saved_var/exact09_{file_insert}_op_embeddings_data.pkl'
file_path2=f'/home/wangc/now/NaviAirway/saved_var/lidc_{file_insert}_op_embeddings_data.pkl'
test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
'LIDC_IDRI_0441',  'LIDC_IDRI_0744', 'LIDC_IDRI_1004','EXACT09_CASE13',
'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']
exact_lidc_concatenated_array , merged_dict, merged_list = load_partial_embeddings(file_path1, file_path2, test_names=test_names)




data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)
device2 = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
# device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


print(X_t.shape)

#需要把数据放到GPU上
cluster_dict={}



# X_t=X_t[200:]
X_t=from_numpy(X_t).float().to(device2)
# from sklearn.cluster import KMeans
num_cluster=2



cluster_labels, cluster_centers = kmeans(
    X=X_t, num_clusters=num_cluster, init=None,distance='euclidean', device=device2
)
print('kmeans is done'+str(num_cluster))



# Assuming X_t and cluster_centers are already on the CPU
X_t_expanded = X_t.unsqueeze(1)  # Dimension becomes (200, 1, 262144)
cluster_centers_expanded = cluster_centers.unsqueeze(0)  # Dimension becomes (1, 2, 262144)

# Move tensors to the GPU
X_t_expanded = X_t_expanded.to(device2)
cluster_centers_expanded = cluster_centers_expanded.to(device2)

batch_size = 50
N = X_t.shape[0]
uncertainy_dict = {}

for i in range(0, N, batch_size):
    # Select a batch of data
    X_batch = X_t_expanded[i:i+batch_size]
    
    # Calculate distances for the batch
    distances_batch = torch.sqrt(torch.sum((X_batch - cluster_centers_expanded) ** 2, dim=2))
    
    # Calculate uncertainty for the batch
    uncertainy_batch = torch.abs(distances_batch[:, 0] - distances_batch[:, 1])
    print(uncertainy_batch.shape)
    # Update uncertainy_dict with batch results
    for j in range(batch_size):
        index = i + j
        if index < N:
            uncertainy_dict[merged_list[index]] = uncertainy_batch[j].cpu().numpy()
    print(f'epoch {i}')
print(len(uncertainy_dict))




file_path = '/home/wangc/now/NaviAirway/saved_var/ae1_uncertainy.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'uncertainy_dict': uncertainy_dict}
    pickle.dump(data_to_save, file)

