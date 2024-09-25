import subprocess
import time
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from torch import from_numpy as from_numpy
from matplotlib import gridspec



#使用示例 show_all_2d_img_with_labels(raw_img_folder,slice_folder,img_num=2000,label_path=label_path)
def show_all_2d_img_with_labels(raw_img_path, output_folder, img_num=None, 
                                num_images_per_batch=16, slice_index=20, label_path=None,raw_img_list=None):
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
                            j=0
                            while(1 not in label_img[j, :, :]):
                                j+=1
                            ax.imshow(raw_img[j, :, :], cmap='gray')
                            ax.contour(label_img[j, :, :], colors='r', linestyles='-')
                        else:
                            ax.imshow(raw_img[slice_index, :, :], cmap='gray')
                            ax.contour(label_img[j, :, :], colors='r', linestyles='-')
                    else:
                        ax.imshow(raw_img[slice_index, :, :], cmap='gray')
                    ax.set_title(f"Image {img_names[index]}\nLabel {label_names[index]} ")
                    ax.axis('off')
               
        # 调整子图之间的间距和布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(os.path.join(output_folder, f"exact09_cluster_{batch_num+1}.png"))
        
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
                                 selected_indices=None,show_index=False):
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
    plt.savefig("visualization.png")
    
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
                    emb3 = embeddings[3].cpu().detach().numpy()
                    embeddings_list.append(emb3)  # 移动到CPU并转为NumPy后添加到列表
                    embeddings_dict[name] = emb3
                    del b
            else:
                    b = from_numpy(raw_img_crop).float()
                    b = b.unsqueeze(0).to(device)
                    embeddings = model.get_embedding(b)
                    emb3 = embeddings[3].cpu().detach().numpy()
                    embeddings_list.append(emb3)  # 移动到CPU并转为NumPy后添加到列表
                    embeddings_dict[name] = emb3
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
