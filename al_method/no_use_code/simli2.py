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

LidcInfo = DatasetInfo("/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128")
LidcInfo.get_case_names("/mnt/wangc/LIDC", "lidc")

Exact09Info = DatasetInfo("/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128")
Exact09Info.get_case_names("/mnt/wangc/EXACT09/EXACT09_3D", "exact09")

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


# file_path1=f'/home/wangc/now/NaviAirway/saved_var/exact09_{file_insert}_op_embeddings_data.pkl'
# file_path2=f'/home/wangc/now/NaviAirway/saved_var/lidc_{file_insert}_op_embeddings_data.pkl'
# with open(file_path1, 'rb') as file:
#     loaded_data = pickle.load(file)
#     exact_embeddings_list = loaded_data['embeddings_list']
#     exact_embeddings_dict = loaded_data['embeddings_dict']
# exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
# print(exact_stacked_embeddings_numpy.shape)
# with open(file_path2, 'rb') as file:
#     loaded_data = pickle.load(file)
#     lidc_embeddings_list = loaded_data['embeddings_list']
#     lidc_embeddings_dict = loaded_data['embeddings_dict']
# lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)
# print(lidc_stacked_embeddings_numpy.shape)
# test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
# 'LIDC_IDRI_0441',  'LIDC_IDRI_0744', 'LIDC_IDRI_1004','EXACT09_CASE13',
# 'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']

# i=0
# new_list=[]
# new_dict={}
# for key,v in exact_embeddings_dict.items():
#     if key[:14] not in test_names:
#         new_list.append(exact_embeddings_list[i])
#         new_dict[key]=exact_embeddings_dict[key]
#     i+=1
# exact_embeddings_list=new_list
# exact_embeddings_dict=new_dict

# i=0
# new_list=[]
# new_dict={}
# for key,v in lidc_embeddings_dict.items():
#     if key[:14] not in test_names:
#         new_list.append(lidc_embeddings_list[i])
#         new_dict[key]=lidc_embeddings_dict[key]
#     i+=1
# lidc_embeddings_list=new_list
# lidc_embeddings_dict=new_dict

# exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
# lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)

# exact_lidc_concatenated_array = np.concatenate((exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy), axis=0)
# merged_dict={**exact_embeddings_dict,**lidc_embeddings_dict}
# merged_list=list(exact_embeddings_dict.keys())+list(lidc_embeddings_dict.keys())
# print(exact_stacked_embeddings_numpy.shape,lidc_stacked_embeddings_numpy.shape,exact_lidc_concatenated_array.shape)


# exact_lidc_concatenated_array = np.concatenate((exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy), axis=0)
# merged_dict={**exact_embeddings_dict,**lidc_embeddings_dict}
# merged_list=raw_case_name_list+lidc_raw_case_name_list


file_path1=f'/home/wangc/now/NaviAirway/saved_var/exact09_{file_insert}_op_embeddings_data.pkl'
file_path2=f'/home/wangc/now/NaviAirway/saved_var/lidc_{file_insert}_op_embeddings_data.pkl'
test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
'LIDC_IDRI_0441',  'LIDC_IDRI_0744', 'LIDC_IDRI_1004','EXACT09_CASE13',
'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']
exact_lidc_concatenated_array , merged_dict, merged_list = load_partial_embeddings(file_path1, file_path2,train_names=None,test_names=test_names)

data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)
device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device3 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


print(X_t.shape)

# 需要把数据放到GPU上
cluster_dict={}


# xi=[('e', 3, 166),
#  ('e', 2, 530),
#  ('e', 7, 162),
#  ('e', 4, 282),
#  ('e', 9, 106),
#  ('l', 403, 93),
#  ('l', 140,295),
#  ('l', 438, 362),
#  ('l', 438, 477),
#  ('l', 529, 283)]


# cu=[('e', 3,418 ),
#  ('e', 2, 467),
#  ('e', 2, 603),
#  ('e', 7, 347),
#  ('l', 438, 547),
#  ('l', 529, 475),
#  ('l', 140, 418),
#  ('l', 403, 98),
#  ('l', 403, 228),
#  ('l', 438, 612)]


# def query_embedding(embeddings_dict, prefix,case_number, patch_number):
#     if case_number < 10:
#         case_number='0'+str(case_number)
#     elif 20<case_number<100:
#         case_number='0'+str(case_number)
#     # if patch_number < 10:
#     #     patch_number='0'+str(patch_number)
#     query = prefix+str(case_number)+'_'+str(patch_number)+'.nii.gz'
#     # print(query)
#     return embeddings_dict[query],query

# query_index={'e':'EXACT09_CASE','l':'LIDC_IDRI_0'}


# # 生成查询并存储在列表中
# query_list1 = []
# query_list2 = []
# query_list3 = []


# three_tuple1=xi[0]

# emb_vector,_=query_embedding(merged_dict,query_index[three_tuple1[0]],three_tuple1[1],three_tuple1[2])

# emb_vectors1=np.zeros((10, *emb_vector.shape))
# emb_vectors2=np.zeros((10, *emb_vector.shape))
# emb_vectors3=np.zeros((10, *emb_vector.shape))

# for i in range(10):

#     emb_vector1, query1 = query_embedding(merged_dict, query_index[xi[i][0]],xi[i][1], xi[i][2])
#     emb_vector2, query2 = query_embedding(merged_dict, query_index[cu[i][0]],cu[i][1], cu[i][2])

#     emb_vectors1[i,:]=emb_vector1
#     query_list1.append(query1)

#     emb_vectors2[i,:]=emb_vector2
#     query_list2.append(query2)


# emb_vectors1_mean=emb_vectors1.mean(axis=0)
# emb_vectors2_mean=emb_vectors2.mean(axis=0)
# center_samples = np.concatenate([emb_vectors1_mean, emb_vectors2_mean], axis=0)
# initial_centers = center_samples.reshape(2, -1)


# X_t=X_t[200:]
X_t=from_numpy(X_t).float().to(device2)
# initial_centers =from_numpy(initial_centers).float().to(device2)

# from sklearn.cluster import KMeans
num_cluster=2


# cluster_labels, cluster_centers = kmeans(
#     X=X_t, num_clusters=num_cluster, init=initial_centers,distance='euclidean', device=device2
# )
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
    
    # Update uncertainy_dict with batch results
    for j in range(batch_size):
        index = i + j
        if index < N:
            uncertainy_dict[merged_list[index]] = uncertainy_batch[j].cpu().numpy()
    print(N)
print('dict',len(uncertainy_dict))


file_path = '/home/wangc/now/NaviAirway/saved_var/ae1_uncertainy_128_data.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'uncertainy_dict': uncertainy_dict}
    pickle.dump(data_to_save, file)
