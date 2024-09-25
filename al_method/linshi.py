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


Precrop_dataset_for_train_path = "/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
print(len(raw_case_name_list))
# raw_case_name_list = os.listdir(Precrop_dataset_for_train_label_path)


lidc_dataset_for_train_path='/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128'
lidc_dataset_for_train_raw_path =lidc_dataset_for_train_path+"/image"
lidc_dataset_for_train_label_path = lidc_dataset_for_train_path+"/label"
lidc_raw_case_name_list = os.listdir(lidc_dataset_for_train_raw_path)
print(len(lidc_raw_case_name_list))



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


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model=SegAirwayModel(in_channels=1, out_channels=2)
model.to(device)
load_pkl='/home/wangc/now/NaviAirway/checkpoint/abc_checkpoint_sample_org_33.pkl'
checkpoint = torch.load(load_pkl)
model.load_state_dict(checkpoint['model_state_dict'])
print(load_pkl)


def process_images1(Precrop_dataset_path,raw_case_name_list,M, N, 
                    model, device,random3dcrop=random3dcrop, normalization=normalization,
                   only_positive=False,#如果仅需要正例请选择
                   need_embedding=3
                   ):
    i =M
    embeddings_list = []  # 用于存储每个图像的 embeddings[3]
    embeddings_dict = {}

    for name in raw_case_name_list:
        if M<=i < N:
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
N=len(lidc_raw_case_name_list)
# N=10
embeddings_list1, embeddings_dict1=process_images1(lidc_dataset_for_train_path,lidc_raw_case_name_list,0, 11760, model, device,
                                                only_positive=True,need_embedding=1
                                                )
# 将列表中的NumPy数组堆叠成一个NumPy数组
stacked_embeddings_numpy1 = np.stack(embeddings_list1, axis=0)
print(stacked_embeddings_numpy1.shape)  # 输出应为 [N, 256, 4, 16, 16]

file_path1 = '/mnt/wangc/NaviAirway/saved_var/lidc_128_op_ae2_encoder1_data_a.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path1), exist_ok=True)

# 保存到文件
with open(file_path1, 'wb') as file:
    data_to_save = {'embeddings_list': embeddings_list1, 'embeddings_dict': embeddings_dict1}
    pickle.dump(data_to_save, file)


Precrop_dataset_for_train_path = "/mnt/wangc/EXACT09/Precrop_dataset_for_EXACT09_128"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
print(len(raw_case_name_list))
# raw_case_name_list = os.listdir(Precrop_dataset_for_train_label_path)


lidc_dataset_for_train_path='/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI_128'
lidc_dataset_for_train_raw_path =lidc_dataset_for_train_path+"/image"
lidc_dataset_for_train_label_path = lidc_dataset_for_train_path+"/label"
lidc_raw_case_name_list = os.listdir(lidc_dataset_for_train_raw_path)
print(len(lidc_raw_case_name_list))



# class Random3DCrop_np(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
#         if isinstance(output_size, int):
#             self.output_size=(output_size, output_size, output_size)
#         else:
#             assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
#             self.output_size=output_size
        
#     def random_crop_start_point(self, input_size):
#         assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
#         d, h, w=input_size
#         d_new, h_new, w_new=self.output_size
        
#         d_new = min(d, d_new)
#         h_new = min(h, h_new)
#         w_new = min(w, w_new)
        
#         assert (d>=d_new and h>=h_new and w>=w_new), "Attention: input size should >= crop size; now, input_size is "+str((d,h,w))+", while output_size is "+str((d_new, h_new, w_new))
        
#         d_start=np.random.randint(0, d-d_new+1)
#         h_start=np.random.randint(0, h-h_new+1)
#         w_start=np.random.randint(0, w-w_new+1)
        
#         return d_start, h_start, w_start
    
#     def __call__(self, img_3d, start_points=None):
#         img_3d=np.array(img_3d)
        
#         d, h, w=img_3d.shape
#         d_new, h_new, w_new=self.output_size
        
#         if start_points == None:
#             start_points = self.random_crop_start_point(img_3d.shape)
        
#         d_start, h_start, w_start = start_points
#         d_end = min(d_start+d_new, d)
#         h_end = min(h_start+h_new, h)
#         w_end = min(w_start+w_new, w)
        
#         crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]
        
#         return crop

# class Normalization_np(object):
#     def __init__(self, windowMin, windowMax):
#         self.name = 'ManualNormalization'
#         assert isinstance(windowMax, (int,float))
#         assert isinstance(windowMin, (int,float))
#         self.windowMax = windowMax
#         self.windowMin = windowMin
    
#     def __call__(self, img_3d):
#         img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
#         img_3d_norm-=np.min(img_3d_norm)
#         max_99_val=np.percentile(img_3d_norm, 99)
#         if max_99_val>0:
#             img_3d_norm = img_3d_norm/max_99_val*255
#         return img_3d_norm
# crop_size = (32, 128, 128)
# windowMin=-1000
# windowMax=150
# random3dcrop=Random3DCrop_np(crop_size)
# normalization=Normalization_np(windowMin, windowMax)


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model=SegAirwayModel(in_channels=1, out_channels=2)
model.to(device)
load_pkl='/home/wangc/now/NaviAirway/checkpoint/abc_checkpoint_sample_org_33.pkl'
checkpoint = torch.load(load_pkl)
model.load_state_dict(checkpoint['model_state_dict'])
print(load_pkl)


def process_images1(Precrop_dataset_path,raw_case_name_list,M, N, 
                    model, device,random3dcrop=random3dcrop, normalization=normalization,
                   only_positive=False,#如果仅需要正例请选择
                   need_embedding=3
                   ):
    i =M
    embeddings_list = []  # 用于存储每个图像的 embeddings[3]
    embeddings_dict = {}

    for name in raw_case_name_list:
        if M<=i < N:
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
N=len(lidc_raw_case_name_list)
# N=10
embeddings_list2, embeddings_dict2=process_images1(lidc_dataset_for_train_path,lidc_raw_case_name_list, 11760,N, model, device,
                                                only_positive=True,need_embedding=1
                                                )
# 将列表中的NumPy数组堆叠成一个NumPy数组
stacked_embeddings_numpy2 = np.stack(embeddings_list1, axis=0)
print(stacked_embeddings_numpy2.shape)  # 输出应为 [N, 256, 4, 16, 16]

file_path2 = '/mnt/wangc/NaviAirway/saved_var/lidc_128_op_ae2_encoder2_data_b.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path2), exist_ok=True)

# 保存到文件
with open(file_path2, 'wb') as file:
    data_to_save = {'embeddings_list': embeddings_list2, 'embeddings_dict': embeddings_dict2}
    pickle.dump(data_to_save, file)



N=len(raw_case_name_list)
# N=10
embeddings_list, embeddings_dict=process_images(Precrop_dataset_for_train_path,raw_case_name_list, N, model, device,
                                                only_positive=True,need_embedding=1
                                                )
# 将列表中的NumPy数组堆叠成一个NumPy数组
stacked_embeddings_numpy = np.stack(embeddings_list, axis=0)
print(stacked_embeddings_numpy.shape)  # 输出应为 [N, 256, 4, 16, 16]

file_path = '/mnt/wangc/NaviAirway/saved_var/exact09_128_op_ae2_encoder1_data.pkl'
# 确保文件夹存在，如果不存在则创建它
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 保存到文件
with open(file_path, 'wb') as file:
    data_to_save = {'embeddings_list': embeddings_list, 'embeddings_dict': embeddings_dict}
    pickle.dump(data_to_save, file)

