import subprocess
import time
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from torch import from_numpy as from_numpy
from matplotlib import gridspec


import pickle
import edt

"""
加载/读取 pkl文件
"""
def save_obj(obj, name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    with open(temp , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    if name[-3:] != 'pkl':
        temp=name+'.pkl'
    else:
        temp=name
    # print(temp)
    with open(temp, 'rb') as f:
        return pickle.load(f)


"""
读取raw_ima_path的文件
"""
#使用示例 show_all_2d_img_with_labels(raw_img_folder,slice_folder,img_num=2000,label_path=label_path)
def show_all_2d_img_with_labels(raw_img_path, 
                                output_folder, 
                                img_num=None, 
                                num_images_per_batch=16, 
                                slice_index=20, 
                                label_path=None,
                                raw_img_list=None,
                                file_name='no_name'):
    if raw_img_list is None:
        raw_img_list = os.listdir(raw_img_path)
    if img_num is None:
        img_num = len(raw_img_list)
    img_num = min(img_num, len(raw_img_list))
    num_batches = (img_num + num_images_per_batch - 1) // num_images_per_batch  # 上取整

    num_rows = 4
    num_cols = 4
    
    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for batch_num in range(num_batches):
        start_index = batch_num * num_images_per_batch
        end_index = min((batch_num + 1) * num_images_per_batch, img_num)
        
        img_list = []
        label_list = []  # 新增：用于存储标签图像

        img_names=[]
        label_names=[]
        
        for i in range(start_index, end_index):
            raw_img_addr = os.path.join(raw_img_path, raw_img_list[i])
            raw_img = io.imread(raw_img_addr, plugin='simpleitk')
            img_list.append(raw_img)

            img_names.append(raw_img_list[i])
            
            if label_path is not None:
                if raw_img_list is None:
                    label_img_list = os.listdir(label_path)
                else:
                    label_img_list=raw_img_list
                label_img_addr = os.path.join(label_path, label_img_list[i])  # 使用相同的索引加载标签图像
                label_img = io.imread(label_img_addr, plugin='simpleitk')
                label_list.append(label_img)  # 存储标签图像

                label_names.append(label_img_list[i])
        
        # 创建一个包含16个子图的图像窗口，使用gridspec布局
        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)
        fig.suptitle(f"Batch {batch_num+1} - Raw Images")
        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                if index < len(img_list):
                    raw_img = img_list[index]
                    ax = fig.add_subplot(gs[i, j])
                    if label_path is not None:
                        label_img = label_list[index]  # 使用相应索引的标签图像
                        if 1 not in label_img[slice_index, :, :]:
                            p=0
                            while(p<label_img.shape[0] and 1 not in label_img[p, :, :]):
                                p+=1
                            if p<label_img.shape[0]:
                                ax.imshow(raw_img[p, :, :], cmap='gray')
                                ax.contour(label_img[p, :, :], colors='r', linestyles='-')
                            else:
                                p=0
                                ax.imshow(raw_img[p, :, :], cmap='gray')
                                ax.contour(label_img[p, :, :], colors='r', linestyles='-')
                                ax.text(1, 1, 'No Label', color='red', fontsize=16, ha='right', va='top', transform=ax.transAxes)
                        else:
                            p=slice_index
                            ax.imshow(raw_img[p, :, :], cmap='gray')
                            ax.contour(label_img[p, :, :], colors='r', linestyles='-')
                    else:
                        p=slice_index
                        ax.imshow(raw_img[p, :, :], cmap='gray')
                    ax.set_title(f"{p} Image {img_names[index]}\nLabel {label_names[index]} ")
                    ax.axis('off')
               
        # 调整子图之间的间距和布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_folder, f"{file_name}_{batch_num+1}.png"))
        
        # 关闭图像窗口，避免重叠
        plt.close()



#图像处理手段，具体使用方法在最后
class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size
        
        d_new = min(d, d_new)
        h_new = min(h, h_new)
        w_new = min(w, w_new)
        
        assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; now, input_size is "+str((d,h,w))+", while output_size is "+str((d_new, h_new, w_new))
        
        d_start=np.random.randint(0, d-d_new+1)
        h_start=np.random.randint(0, h-h_new+1)
        w_start=np.random.randint(0, w-w_new+1)
        
        return d_start, h_start, w_start
    
    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        d_start, h_start, w_start = start_points
        d_end = min(d_start+d_new, d)
        h_end = min(h_start+h_new, h)
        w_end = min(w_start+w_new, w)
        
        crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return crop

class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        return img_3d_norm
crop_size = (32, 128, 128)
windowMin=-1000
windowMax=150
random3dcrop=Random3DCrop_np(crop_size)
normalization=Normalization_np(windowMin, windowMax)



# generate_folder_for_selected(source_folder, target_folder, names_filtered, num=50)
def generate_folder_for_selected(source, #原数据集位置
                                 target, #目的路径
                                 selected_name, #要进行选择的name
                                 num=None): #如果只是想看一部分，而不是选择全部，设置num即可
    """将原数据集的一部分文件取出，放入目的文件夹中"""
    
    address1 = ["/image", "/label"]

    for adr in address1:
        source_path = source+adr
        target_path = target+adr
        os.makedirs(target_path, exist_ok=True)
        if num is None:# 如果 num 未指定，默认为选定名称的长度
            num=len(selected_name)
        for i in range(num):
            print(f"{i/num*100:.2f}%", end="\r")
            time.sleep(0.5)  # 模拟耗时操作
            file_name = selected_name[i]
            source_file = os.path.join(source_path, file_name)
            target_file = os.path.join(target_path, file_name)  
            subprocess.run(['cp', source_file, target_file])## 复制文件到目标路径



# class_0_indices,class_0_names_filtere = visualize_and_return_indices(X_embedded_2d, cluster_labels, embeddings_dict,selected_indices=class_0_indices)
def visualize_and_return_indices(X_embedded_2d, cluster_labels, embeddings_dict, 
                                 x1=100,y1=100,x2=None,y2=None,
                                 selected_indices=None,show_index=False,save_path=None):
#根据X_embedded_2d、cluster_labels、embeddings_dict将得到的聚类图绘出
#会去除一些值过于极端的点，具体和x1，x2，y1，y2有关
#此外还可以根据selected_indices，仅将需要选择点 在图像中绘出
#show_index为True时，将对应点的索引也写出来

#return值 filtered_indices表示这次选择了哪些点, names_filtered表示这些点的索引
    if x2 is None and y2 is None:
        filtered_indices = np.where((abs(X_embedded_2d[:, 0]) < abs(x1)) & (abs(X_embedded_2d[:, 1]) < abs(y1)))[0]
    
    else:

        # 创建布尔索引
        x_condition = (X_embedded_2d[:, 0] >= x1) & (X_embedded_2d[:, 0] <= x2)
        y_condition = (X_embedded_2d[:, 1] >= y1) & (X_embedded_2d[:, 1] <= y2)#纵轴
    
        # 通过两个条件的逻辑与来筛选元素
        filtered_indices = np.where(x_condition & y_condition)[0]
    print('filtered_indices',len(filtered_indices))
    if selected_indices is not None:
        intersection_set = set(filtered_indices).intersection(selected_indices)
        filtered_indices = list(intersection_set)
        print('selected_indices',len(filtered_indices))
    # 筛选出符合阈值的数据
    filtered_embeddings = X_embedded_2d[filtered_indices]
    

    # 提取嵌入坐标和对应的名称
    names = list(embeddings_dict.keys())
    names_filtered = [names[i] for i in filtered_indices]
    unique_labels = np.unique(cluster_labels[filtered_indices])
    cmap = plt.get_cmap('viridis', len(unique_labels))
    colors = [cmap(i) for i in range(len(unique_labels))]
    custom_cmap = ListedColormap(colors)
    
    # 清除之前的图像，以防止重叠
    plt.clf()
    
    # 绘制图像
    plt.figure(figsize=(8, 6), dpi=100)
    if show_index is True:
        for i, j in enumerate(filtered_indices):
            plt.annotate(j, (filtered_embeddings[i, 0], filtered_embeddings[i, 1]), fontsize=8, alpha=0.7)
        # print(i)



    num_classes = len(unique_labels)
    random_colors = np.random.rand(num_classes, 3)
    custom_cmap = ListedColormap(random_colors)

    plt.scatter(filtered_embeddings[:, 0], filtered_embeddings[:, 1], c=cluster_labels[filtered_indices], cmap=custom_cmap)
    print('filtered_embeddings.shape',filtered_embeddings.shape)
    plt.title("t-SNE 2D Embedding with Cluster Coloring (Perplexity=70)")
    # sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=cluster_labels.min(), vmax=cluster_labels.max()))
    # sm._A = []  # 设置一个空的数组
    
    # 保存图片
    if save_path is None:
        save_path="visualization.png"
    plt.savefig(save_path)
    
    # 返回结果
    return filtered_indices, names_filtered



def analysis_cluster(label_path,raw_case_name_list=None,slice_index=None):
    def get_label_airway_pixels(label_img, slice_index=None):
        if slice_index is not None:
            if 1 not in label_img[slice_index, :, :]:
                j = 0
                while j < label_img.shape[0] and 1 not in label_img[j, :, :]:
                    j += 1
                if j >= label_img.shape[0]:
                    j = j - 1
                return label_img[j, :, :].sum()
            else:
                return label_img[slice_index, :, :].sum()
        else:
            return label_img.sum()
        

    if raw_case_name_list is None:
        label_img_list = os.listdir(label_path)
    else:
        label_img_list = raw_case_name_list
    pixels_num_list = []
    
    for name in label_img_list:
        label_img_addr = os.path.join(label_path, name)
        label_img = io.imread(label_img_addr, plugin='simpleitk')
        pixels_num = get_label_airway_pixels(label_img, slice_index)
        pixels_num_list.append(pixels_num)

    # 使用numpy计算均值和标准差
    mean_value = np.mean(pixels_num_list)
    std_deviation = np.std(pixels_num_list)
    num_sample=len(label_img_list)
    return mean_value, std_deviation,num_sample  # 均值和标准差


def process_images(Precrop_dataset_path,raw_case_name_list, N, 
                    model, device,random3dcrop=random3dcrop, normalization=normalization,
                   only_positive=False,#如果仅需要正例请选择
                   need_embedding=3
                   ):
    i = 0
    embeddings_list = []  # 用于存储每个图像的 embeddings[3]
    embeddings_dict = {}

    for name in raw_case_name_list:
        if i < N:
            i += 1
            img_addr = Precrop_dataset_path +"/image"+ "/" + name
            print(f'this is {i}')
            img = io.imread(img_addr, plugin='simpleitk')

            start_points = random3dcrop.random_crop_start_point(img.shape)  # 起点
            raw_img_crop = random3dcrop(np.array(img, float), start_points=start_points)
            raw_img_crop = normalization(raw_img_crop)
            raw_img_crop = np.expand_dims(raw_img_crop, axis=0)

            if only_positive is True:
                label_addr=Precrop_dataset_path +"/label"+ "/" + name
                label_img=io.imread(label_addr, plugin='simpleitk')
                if 1 in label_img:
                    b = from_numpy(raw_img_crop).float()
                    b = b.unsqueeze(0).to(device)
                    embeddings = model.get_embedding(b)
                    if 0<=need_embedding<len(embeddings):
                        emb = embeddings[need_embedding].cpu().detach().numpy()
                    else:
                        emb=model(b).cpu().detach().numpy()
                    embeddings_list.append(emb)  # 移动到CPU并转为NumPy后添加到列表
                    embeddings_dict[name] = emb
                    del b
            else:
                    b = from_numpy(raw_img_crop).float()
                    b = b.unsqueeze(0).to(device)
                    embeddings = model.get_embedding(b)
                    if 0<=need_embedding<len(embeddings):
                        emb = embeddings[need_embedding].cpu().detach().numpy()
                    else:
                        emb=model(b).cpu().detach().numpy()
                    embeddings_list.append(emb)  # 移动到CPU并转为NumPy后添加到列表
                    embeddings_dict[name] = emb
                    del b

    return embeddings_list, embeddings_dict
#使用gpu加速的同一方法
# # intial shape torch.Size([1, 1, 32, 128, 128])



# batch_size = 8 # 设置批处理大小
# i = 0
# N = 2000
# embeddings_list = []  # 用于存储每个图像的 embeddings[3]

# batch = []  # 用于存储当前批次的图像
# for name in raw_case_name_list:
#     if i < N:

#         img_addr = Precrop_dataset_for_train_label_path + "/" + name
#         i=i+1
#         img = io.imread(img_addr, plugin='simpleitk')

#         start_points = random3dcrop.random_crop_start_point(img.shape)  # 起点
#         raw_img_crop = random3dcrop(np.array(img, float), start_points=start_points)
#         raw_img_crop = normalization(raw_img_crop)
#         raw_img_crop = np.expand_dims(raw_img_crop, axis=0)

#         batch.append(from_numpy(raw_img_crop).float().to(device))  # 将数据移到 GPU

#         if len(batch) == batch_size:
#             print(i/batch_size)
#             batch_tensor = torch.cat(batch, dim=0)
#             batch_tensor = torch.unsqueeze(batch_tensor, 1)
#             print(batch_tensor.shape)
#             embeddings_batch = model.get_embedding(batch_tensor)
#             embeddings_list.extend([emb.cpu().detach().numpy() for emb in embeddings_batch[3]])  # 移动到CPU并转为NumPy后添加到列表
#             del batch[:]  # 清空批次数据

# # 处理剩余的未满批次数据
# if len(batch) > 0:
#     batch_tensor = torch.cat(batch, dim=0)
#     embeddings_batch = model.get_embedding(batch_tensor)
#     embeddings_list.extend([emb.cpu().detach().numpy() for emb in embeddings_batch[3]])

# # 将列表中的NumPy数组堆叠成一个NumPy数组
# stacked_embeddings_numpy = np.stack(embeddings_list, axis=0)
# print(stacked_embeddings_numpy.shape)  # 输出应为 [N, 256, 4, 16, 16]

# # 现在您可以将 stacked_embeddings_numpy 存储到磁盘或进行其他操作



import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans as SKLearnKMeans

#change from kmeans_pytorch https://github.com/subhadarship/kmeans_pytorch
def initialize(X, num_clusters, init=None, random_state=None):

    if init is None:
        num_samples = len(X)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
        initial_state = X[indices]
    elif torch.is_tensor(init):
        if init.shape[0] != num_clusters or init.shape[1] != X.shape[1]:
            raise ValueError("The shape of the custom tensor 'init' should be (num_clusters, num_features).")
        initial_state = init.to(X.device)
    else:
        raise ValueError("Invalid type for 'init'. Use 'str' for predefined methods or 'torch.tensor' for custom initialization.")
    print(initial_state.shape)
    return initial_state


def kmeans(
    X,
    num_clusters,
    distance='euclidean',
    tol=1e-4,
    init=None,
    max_iter=300,
    device=torch.device('cpu'),
    random_state=None
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param init: (str or torch.tensor) initialization method [options: 'k-means++', 'random', or custom tensor]
                 [default: 'k-means++']
    :param max_iter: (int) maximum number of iterations [default: 300]
    :param device: (torch.device) device [default: cpu]
    :param random_state: (int or RandomState) seed or RandomState for reproducibility [default: None]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters, init=init, random_state=random_state)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol or iteration >= max_iter:
            break

    return choice_cluster.cpu(), initial_state.cpu()
def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def load_partial_embeddings(file_path1, file_path2, train_names=None,test_names=None):
    with open(file_path1, 'rb') as file:
        loaded_data = pickle.load(file)
        exact_embeddings_list = loaded_data['embeddings_list']
        exact_embeddings_dict = loaded_data['embeddings_dict']

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)

    with open(file_path2, 'rb') as file:
        loaded_data = pickle.load(file)
        lidc_embeddings_list = loaded_data['embeddings_list']
        lidc_embeddings_dict = loaded_data['embeddings_dict']
    # print('exact',len(exact_embeddings_dict),'lidc',len(lidc_embeddings_dict))
    lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)
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
    for key, v in lidc_embeddings_dict.items():
        if key[:14] in train_names or key in train_names:
            new_list.append(lidc_embeddings_list[i])
            new_dict[key] = lidc_embeddings_dict[key]
        i += 1
    lidc_embeddings_list = new_list
    lidc_embeddings_dict = new_dict

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
    lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)

    exact_lidc_concatenated_array = np.concatenate((exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy), axis=0)
    merged_dict = {**exact_embeddings_dict, **lidc_embeddings_dict}
    merged_list = list(exact_embeddings_dict.keys()) + list(lidc_embeddings_dict.keys())

    return exact_lidc_concatenated_array, merged_dict, merged_list

def select_from_candidates(select_num,candidates_array, candidates_dict , candidates_list,device):
    def sim(a, b):
        # 调整输入张量的维度
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)

        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return cos_sim

    def get_score2(i, score1s, score2s):

        # 将 float('-inf') 的数据类型设置为与 score1s 和 score2s 一致
        negative_inf = torch.tensor(float('-inf'), dtype=score1s.dtype, device=score1s.device)
        # 计算score2s，保持已经是负无穷大的值不变
        new_score2s = torch.where(score1s == negative_inf, negative_inf, (i + 1) / score1s)
        # 检查是否有 score2s 中的值已经是负无穷大，如果是，则保持不变
        score2s = torch.where(score2s == negative_inf, negative_inf, new_score2s)
        # 获取最大值对应的索引
        new_candidate = torch.argmax(score2s).item()
        return score2s, new_candidate
    
    candidates_tensor=from_numpy(candidates_array)
    candidates_tensor = candidates_tensor.to(device)

    scores_shape = candidates_tensor.shape[0]
    score1s = torch.ones(scores_shape, device=candidates_tensor.device)
    score2s = torch.zeros(scores_shape, device=candidates_tensor.device)
    selcet_list=[]

    new_candidate = 0
    score2s[new_candidate] = float('-inf')
    new_candidate_representiveness = candidates_tensor[new_candidate]
    selcet_list.append(candidates_list[new_candidate])
    for i in range(select_num-1):
        print(f'epoch {i}')
        # 计算相似度
        sim_output = sim(new_candidate_representiveness, candidates_tensor)
        # print(sim_output.shape)
        score1s = score1s + sim_output

        # 调用你的 get_score2 函数
        score2s, new_candidate = get_score2(i, score1s, score2s)
        # print('new_candidate',new_candidate,candidates_list[new_candidate])
        selcet_list.append(candidates_list[new_candidate])
        score2s[new_candidate] = float('-inf')
        new_candidate_representiveness = candidates_tensor[new_candidate]

    return selcet_list