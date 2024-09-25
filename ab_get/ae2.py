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

file_path1='/mnt/wangc/NaviAirway/saved_var/exact09_128_op_ae2_encoder2_data.pkl'
file_path2='/mnt/wangc/NaviAirway/saved_var/lidc_128_op_ae2_encoder2_data_a.pkl'
file_path3='/mnt/wangc/NaviAirway/saved_var/lidc_128_op_ae2_encoder2_data_b.pkl'
test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
'LIDC_IDRI_0441',  'LIDC_IDRI_0744', 'LIDC_IDRI_1004','EXACT09_CASE13',
'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']
def load_partial_embeddings1(file_path1, file_path2, file_path3,train_names=None,test_names=None):
    with open(file_path1, 'rb') as file:
        loaded_data = pickle.load(file)
        exact_embeddings_list = loaded_data['embeddings_list']
        exact_embeddings_dict = loaded_data['embeddings_dict']

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)

    with open(file_path2, 'rb') as file:
        loaded_data = pickle.load(file)
        lidc_embeddings_list1 = loaded_data['embeddings_list']
        lidc_embeddings_dict1 = loaded_data['embeddings_dict']
    # print('exact',len(exact_embeddings_dict),'lidc',len(lidc_embeddings_dict))
    lidc_stacked_embeddings_numpy1 = np.stack(lidc_embeddings_list1, axis=0)
    with open(file_path3, 'rb') as file:
        loaded_data = pickle.load(file)
        lidc_embeddings_list2 = loaded_data['embeddings_list']
        lidc_embeddings_dict2 = loaded_data['embeddings_dict']
    # print('exact',len(exact_embeddings_dict),'lidc',len(lidc_embeddings_dict))
    lidc_stacked_embeddings_numpy2 = np.stack(lidc_embeddings_list2, axis=0)
    names=['EXACT09_CASE01', 'EXACT09_CASE02', 'EXACT09_CASE03',
       'EXACT09_CASE04', 'EXACT09_CASE05', 'EXACT09_CASE06',
       'EXACT09_CASE07', 'EXACT09_CASE08', 'EXACT09_CASE09',
       'EXACT09_CASE10', 'EXACT09_CASE11', 'EXACT09_CASE12',
       'EXACT09_CASE13', 'EXACT09_CASE14', 'EXACT09_CASE15',
       'EXACT09_CASE16', 'EXACT09_CASE17', 'EXACT09_CASE18',
       'EXACT09_CASE19', 'EXACT09_CASE20', 'LIDC_IDRI_0066',
       'LIDC_IDRI_0140', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
       'LIDC_IDRI_0403', 'LIDC_IDRI_0430', 'LIDC_IDRI_0438',
       'LIDC_IDRI_0441', 'LIDC_IDRI_0490', 'LIDC_IDRI_0529',
       'LIDC_IDRI_0606', 'LIDC_IDRI_0621', 'LIDC_IDRI_0648',
       'LIDC_IDRI_0651', 'LIDC_IDRI_0657', 'LIDC_IDRI_0663',
       'LIDC_IDRI_0673', 'LIDC_IDRI_0676', 'LIDC_IDRI_0684',
       'LIDC_IDRI_0696', 'LIDC_IDRI_0698', 'LIDC_IDRI_0710',
       'LIDC_IDRI_0722', 'LIDC_IDRI_0744', 'LIDC_IDRI_0757',
       'LIDC_IDRI_0778', 'LIDC_IDRI_0784', 'LIDC_IDRI_0810',
       'LIDC_IDRI_0813', 'LIDC_IDRI_0819', 'LIDC_IDRI_0831',
       'LIDC_IDRI_0837', 'LIDC_IDRI_0856', 'LIDC_IDRI_0874',
       'LIDC_IDRI_0876', 'LIDC_IDRI_0909', 'LIDC_IDRI_0920',
       'LIDC_IDRI_0981', 'LIDC_IDRI_1001', 'LIDC_IDRI_1004']
    if train_names is None and test_names is None:
        assert False
    if train_names is None:
        train_names = [name for name in names if name not in test_names]
    i = 0
    new_list = []
    new_dict = {}
    for key, v in exact_embeddings_dict.items():
        # print(key,key[:14])
        if key[:14] in train_names or key in train_names:
            new_list.append(exact_embeddings_list[i])
            new_dict[key] = exact_embeddings_dict[key]
        i += 1
    exact_embeddings_list = new_list
    exact_embeddings_dict = new_dict

    i = 0
    new_list = []
    new_dict = {}
    for key, v in lidc_embeddings_dict1.items():
        if key[:14] in train_names or key in train_names:
            new_list.append(lidc_embeddings_list1[i])
            new_dict[key] = lidc_embeddings_dict1[key]
        i += 1
    lidc_embeddings_list1 = new_list
    lidc_embeddings_dict1 = new_dict
    
    i = 0
    new_list = []
    new_dict = {}
    for key, v in lidc_embeddings_dict2.items():
        if key[:14] in train_names or key in train_names:
            new_list.append(lidc_embeddings_list2[i])
            new_dict[key] = lidc_embeddings_dict2[key]
        i += 1
    lidc_embeddings_list2 = new_list
    lidc_embeddings_dict2= new_dict

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
    lidc_stacked_embeddings_numpy1 = np.stack(lidc_embeddings_list1, axis=0)
    lidc_stacked_embeddings_numpy2 = np.stack(lidc_embeddings_list2, axis=0)

    exact_lidc_concatenated_array = np.concatenate((exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy1, lidc_stacked_embeddings_numpy2 ), axis=0)
    merged_dict = {**exact_embeddings_dict, **lidc_embeddings_dict1, **lidc_embeddings_dict2}
    merged_list = list(exact_embeddings_dict.keys()) + list(lidc_embeddings_dict1.keys())+list(lidc_embeddings_dict2.keys())

    return exact_lidc_concatenated_array, merged_dict, merged_list
# exact_lidc_concatenated_array , merged_dict, merged_list = load_partial_embeddings1(file_path1, file_path2, file_path3,test_names=test_names)


file_path1='/mnt/wangc/NaviAirway/saved_var/exact09_128_op_segmentation_data.pkl'
file_path2='/mnt/wangc/NaviAirway/saved_var/lidc_128_op_segmentation_data.pkl'
exact_lidc_concatenated_array , merged_dict, merged_list = load_partial_embeddings(file_path1, file_path2,test_names=test_names)



data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)
device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device3 = torch.device( 'cpu')
data_shape = exact_lidc_concatenated_array.shape

# device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


print(X_t.shape)

#需要把数据放到GPU上
cluster_dict={}



# X_t=X_t[200:]
X_t=from_numpy(X_t).float().to(device3)
# from sklearn.cluster import KMeans
num_cluster=2



cluster_labels, cluster_centers = kmeans(
    X=X_t, num_clusters=num_cluster, init=None,distance='euclidean', device=device3
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
    X_t_batch = torch.from_numpy(X_t[i:i+batch_size]).float().to(device2)
    X_batch = X_t_batch.unsqueeze(1)  # Dimension becomes (batch_size, 1, 262144)
    
    
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




file_path = '/home/wangc/now/NaviAirway/saved_var/ae2_uncertainy_encoder5.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'uncertainy_dict': uncertainy_dict}
    pickle.dump(data_to_save, file)

