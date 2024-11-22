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
from active_utils.embedding_tools import get_embeddings
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


# 从文件加载样本的特征表示——————————————————————————

# file_path1='/home/wangc/now/NaviAirway/saved_var/exact09_256_embeddings_data.pkl'
# file_path2='/home/wangc/now/NaviAirway/saved_var/lidc_256_embeddings_data.pkl'

file_path1='/home/wangc/now/NaviAirway/saved_var/exact09_128_op_embeddings_data.pkl'
file_path2='/home/wangc/now/NaviAirway/saved_var/lidc_128_op_embeddings_data.pkl'
with open(file_path1, 'rb') as file:
    loaded_data = pickle.load(file)
    exact_embeddings_list = loaded_data['embeddings_list']
    exact_embeddings_dict = loaded_data['embeddings_dict']
exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)

with open(file_path2, 'rb') as file:
    loaded_data = pickle.load(file)
    lidc_embeddings_list = loaded_data['embeddings_list']
    lidc_embeddings_dict = loaded_data['embeddings_dict']
lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)


##剔除验证集样本——————————————————————————
test_names = ['LIDC_IDRI_0066', 'LIDC_IDRI_0328', 'LIDC_IDRI_0376',
'LIDC_IDRI_0441', 'EXACT09_CASE13', 'LIDC_IDRI_0744', 'LIDC_IDRI_1004',
'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']

i=0
new_list=[]
new_dict={}
for key,v in exact_embeddings_dict.items():
    if key[:14] not in test_names:
        new_list.append(exact_embeddings_list[i])
        new_dict[key]=exact_embeddings_dict[key]
    i+=1
exact_embeddings_list=new_list
exact_embeddings_dict=new_dict

i=0
new_list=[]
new_dict={}
for key,v in lidc_embeddings_dict.items():
    if key[:14] not in test_names:
        new_list.append(lidc_embeddings_list[i])
        new_dict[key]=lidc_embeddings_dict[key]
    i+=1
lidc_embeddings_list=new_list
lidc_embeddings_dict=new_dict


##合并两个数据集——————————————————————————
exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)

exact_lidc_concatenated_array = np.concatenate((exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy), axis=0)
merged_dict={**exact_embeddings_dict,**lidc_embeddings_dict}
merged_list=list(exact_embeddings_dict.keys())+list(lidc_embeddings_dict.keys())
print(exact_stacked_embeddings_numpy.shape,lidc_stacked_embeddings_numpy.shape,exact_lidc_concatenated_array.shape)


data_shape = exact_lidc_concatenated_array.shape

X_t = exact_lidc_concatenated_array.reshape(data_shape[0], -1)

device2 = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


##聚类中心生成——————————————————————————
xi=[('e', 3, 166),
 ('e', 2, 530),
 ('e', 7, 162), 
 ('e', 4, 282),
 ('e', 9, 106), 
 ('l', 403, 93), 
 ('l', 140,295), 
 ('l', 438, 362), 
 ('l', 438, 477), 
 ('l', 529, 283)]


cu=[('e', 3, 418), 
 ('e', 2, 467),
 ('e', 2, 603),
 ('e', 7, 347),
 ('l', 438, 547),
 ('l', 529, 475),
 ('l', 140, 418),
 ('l', 403, 98), 
 ('l', 403, 228),
 ('l', 438, 612)]

# xi=[('e', 3, 17),('e', 2, 7),('e', 1, 24),('e', 4, 21),('e', 8, 8),('l', 66, 36),('l', 328,50),('l', 328, 22),('l', 376, 22),('l', 722, 26)]
# cu=[('e', 1, 52),('e', 2, 55),('e', 2, 70),('e', 3, 77),('l', 140, 55),('l', 403, 21),('l', 698, 34),('l', 744, 46),('l', 403, 37),('l', 673, 69)]

def query_embedding(embeddings_dict, prefix,case_number, patch_number):
    if case_number < 10:
        case_number='0'+str(case_number)
    elif 20<case_number<100:
        case_number='0'+str(case_number)
    # if patch_number < 10:
    #     patch_number='0'+str(patch_number)
    query = prefix+str(case_number)+'_'+str(patch_number)+'.nii.gz'
    # print(query)
    return embeddings_dict[query],query

query_index={'e':'EXACT09_CASE','l':'LIDC_IDRI_0'}


# 生成查询并存储在列表中
query_list1 = []
query_list2 = []
query_list3 = []


three_tuple1=xi[0]    

emb_vector,_=query_embedding(merged_dict,query_index[three_tuple1[0]],three_tuple1[1],three_tuple1[2])

emb_vectors1=np.zeros((10, *emb_vector.shape))
emb_vectors2=np.zeros((10, *emb_vector.shape))
# emb_vectors3=np.zeros((10, *emb_vector.shape))

for i in range(10):

    emb_vector1, query1 = query_embedding(merged_dict, query_index[xi[i][0]],xi[i][1], xi[i][2])
    emb_vector2, query2 = query_embedding(merged_dict, query_index[cu[i][0]],cu[i][1], cu[i][2])
    # print(query2)
    emb_vectors1[i,:]=emb_vector1
    query_list1.append(query1)

    emb_vectors2[i,:]=emb_vector2
    query_list2.append(query2)


emb_vectors1_mean=emb_vectors1.mean(axis=0)
emb_vectors2_mean=emb_vectors2.mean(axis=0)
center_samples = np.concatenate([emb_vectors1_mean, emb_vectors2_mean], axis=0)
initial_centers = center_samples.reshape(2, -1)


X_t=from_numpy(X_t).float().to(device2)
initial_centers =from_numpy(initial_centers).float().to(device2)


##进行聚类——————————————————————————
num_clusters=[2]
cluster_dict={}
for num_cluster in num_clusters:
    print('正在处理'+str(num_cluster))

    cluster_labels, cluster_centers = kmeans(
        X=X_t, num_clusters=num_cluster, init=initial_centers,distance='euclidean', device=device2
    )

    # cluster_labels, cluster_centers = kmeans(
    #     X=X_t, num_clusters=num_cluster, init=None,distance='euclidean', device=device2
    # )




    # 将cluster_labels从CUDA设备复制到CPU上
    cluster_labels = cluster_labels.cpu()


    # print(X_t[0].shape)



    # 将torch.Tensor转换为NumPy数组
    X_t_with_labels = np.column_stack((X_t.cpu().numpy(), cluster_labels.reshape(-1, 1)))

    # X_t_with_labels = np.column_stack((X_t, cluster_labels.reshape(-1, 1)))

    N=X_t.shape[0]
    # 使用 perplexity 为 50 的 t-SNE 进行嵌入

##绘制tsne图——————————————————————————

    tsne = TSNE(n_components=2, perplexity=70)
    X_embedded_2d = tsne.fit_transform(X_t_with_labels)
    embedding_dict = {merged_list[i]: X_embedded_2d[i] for i in range(N)}
    # center1,center2=cluster_centers[0],cluster_centers[1]




    # 使用 t-SNE 进行降维
    threshold_dim1 = 20000  # 调整阈值
    threshold_dim2 = 20000  # 调整阈值

    # 筛选出维度值在一定范围内的数据
    filtered_indices = np.where((abs(X_embedded_2d[:, 0]) < threshold_dim1) & (abs(X_embedded_2d[:, 1]) < threshold_dim2))[0]

    # 筛选出符合阈值的数据
    filtered_embeddings = X_embedded_2d[filtered_indices]


    # 绘制二维散点图，只绘制符合阈值的数据
    # 可视化二维嵌入并根据聚类标签着色
    # colors = list(mcolors.TABLEAU_COLORS.values())
    # custom_cmap = mcolors.ListedColormap(colors)


    unique_labels = np.unique(cluster_labels)
    cmap = plt.get_cmap('viridis', len(unique_labels))
    colors = [cmap(i) for i in range(len(unique_labels))]

    query1_points = [True if name in query_list1 else False for name in embedding_dict.keys()]
    query2_points = [True if name in query_list2 else False for name in embedding_dict.keys()]
    # query3_points = [True if name in query_list3 else False for name in embeddings_dict.keys()]
    # # 使用 ListedColormap 为不同的聚类簇指定颜色
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(colors)
    # 创建图例标签
    legend_labels = [f'class{label}' for label in np.unique(cluster_labels)]

    # 创建图例句柄
    legend_handles = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(unique_labels))]

    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1], c=cluster_labels[filtered_indices], cmap=custom_cmap)
    plt.title("t-SNE 2D Embedding with Cluster Coloring (Perplexity=70)")

    # 
    # 添加图例
    plt.legend(handles=legend_handles, loc='upper right')
    m1=0
    m2=0
    # # 添加大黑点和标记
    for i in range(len(filtered_embeddings)):
        if query1_points[filtered_indices[i]] is True:
            m1+=1
            plt.scatter(filtered_embeddings[i, 0], filtered_embeddings[i, 1], c='red', s=150, marker='o', label='query_list1') 
        if query2_points[filtered_indices[i]] is True:
            m2+=1
            plt.scatter(filtered_embeddings[i, 0], filtered_embeddings[i, 1], c='pink', s=150, marker='o', label='query_list2') 
    #     # if query3_points[filtered_indices[i]] is True:
    #     #     plt.scatter(filtered_embeddings[i, 0], filtered_embeddings[i, 1], c='orange', s=150, marker='o', label='query_list3')  

    # print(m1,m2)
    # save_path='/home/wangc/now/NaviAirway/saved_picture/for_128_cluster/cluster_'+str(num_cluster)+'.png'
    save_path='/home/wangc/now/NaviAirway/saved_picture/for_128_cluster/cluster_set.png'

    plt.savefig(save_path)




    # saved_name='cluster'+str(num_cluster)+'.png'
    saved_name='cluster_set2.png'



##保存聚类信息——————————————————————————
    all_indices,_ = visualize_and_return_indices(X_embedded_2d, cluster_labels, merged_dict,
                                    x1=100,y1=100,x2=None,y2=None,selected_indices=None,save_path='/home/wangc/now/NaviAirway/saved_picture/for_128_cluster/'+saved_name)



    indices_dict={}
    for i in range(num_cluster):
        this_class='class_'+str(i)
        indices = np.where(cluster_labels == i)[0]
        class_i_indices,class_i_names_filtered = visualize_and_return_indices(X_embedded_2d, cluster_labels, merged_dict,
        selected_indices=indices)
        indices_dict[this_class]=class_i_names_filtered
    cluster_dict['give_init']=indices_dict 
save_obj(cluster_dict, "/home/wangc/now/NaviAirway/saved_objs/for_128_objs/set_indices")
