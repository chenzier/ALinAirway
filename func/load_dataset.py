# dataset load and transform
import numpy as np
import skimage.io as io
import random
import h5py


from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio
class Random3DCrop_np2(object):
    def __init__(self, output_size):
        assert isinstance(
            output_size, (int, tuple)
        ), "Attention: random 3D crop output size: an int or a tuple (length:3)"
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert (
                len(output_size) == 3
            ), "Attention: random 3D crop output size: a tuple (length:3)"
            self.output_size = output_size

    def random_crop_start_point(self, img_3d):
        """Find a random start point within the foreground."""
        assert len(img_3d.shape) == 3, "Attention: input must be a 3D image"

        d, h, w = img_3d.shape
        d_new, h_new, w_new = self.output_size

        # Ensure crop size does not exceed image dimensions
        d_new = min(d, d_new)
        h_new = min(h, h_new)
        w_new = min(w, w_new)

        # Locate all foreground points (value == 1)
        foreground_points = np.argwhere(img_3d == 1)
        if foreground_points.size == 0:
            raise ValueError("No foreground (value == 1) found in the input image.")

        # Randomly select a foreground point
        center_point = foreground_points[np.random.choice(len(foreground_points))]
        d_center, h_center, w_center = center_point

        # Calculate valid start ranges to ensure crop fits within image dimensions
        d_start = max(0, min(d - d_new, d_center - d_new // 2))
        h_start = max(0, min(h - h_new, h_center - h_new // 2))
        w_start = max(0, min(w - w_new, w_center - w_new // 2))

        return d_start, h_start, w_start

    def __call__(self, img_3d, start_points=None):
        img_3d = np.array(img_3d)

        d, h, w = img_3d.shape
        d_new, h_new, w_new = self.output_size

        if start_points is None:
            start_points = self.random_crop_start_point(img_3d)

        d_start, h_start, w_start = start_points
        d_end = min(d_start + d_new, d)
        h_end = min(h_start + h_new, h)
        w_end = min(w_start + w_new, w)

        crop = img_3d[d_start:d_end, h_start:h_end, w_start:w_end]

        return crop


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

class airway_dataset(Dataset):
    # train_dataset_more_focus_on_airways_of_low_generation = airway_dataset(dataset_info_more_focus_on_airways_of_low_generation)
    def __init__(self, data_dict, num_of_samples = None):
        # each item of data_dict is {name:{"image": img path, "label": label path}}
        self.data_dict = data_dict
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            chosen_names = np.random.choice(np.array(list(data_dict)), num_of_samples, replace=False)
        else:#只执行这个
            chosen_names = np.array(list(data_dict))
        # print(num_of_samples is None)#True
        # print(list(data_dict))#生成一个list，只包括这个字典的keys
        # print(chosen_names)#将上面的list(data_dict转为<class 'numpy.ndarray'>

        self.name_list = chosen_names
        # print('len',len(self.name_list))#36290 27535
        self.para = {}
        self.set_para()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # print('idx{}',idx)#idx{} 11979 看得出是随机生成的
        # 下面调用的get函数,返回的是一个字典{'image':<class 'torch.Tensor'>&shape=torch.Size([1, 32, 128, 128]),
        #                                'label':<class 'torch.Tensor'>&shape=torch.Size([1, 32, 128, 128]),}
        # 并且数据经过了标准化、裁剪等处理
        return self.get(idx, file_format=self.para["file_format"],\
            crop_size=self.para["crop_size"],\
                windowMin=self.para["windowMin"],\
                    windowMax=self.para["windowMax"],\
                        need_tensor_output=self.para["need_tensor_output"],\
                            need_transform=self.para["need_transform"])
    # 全部默认
    def set_para(self, file_format='.nii.gz', crop_size=32, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    # 看了下参数全默认,只有idx是从getitem迭代得到的
    def get(self, idx, file_format='.nii.gz', crop_size=32, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):

        random3dcrop=Random3DCrop_np(crop_size)
        normalization=Normalization_np(windowMin, windowMax)

        name = self.name_list[idx]

        raw_img = io.imread(self.data_dict[name]["image"], plugin='simpleitk')
        label_img = io.imread(self.data_dict[name]["label"], plugin='simpleitk')
        # 参数可选npy/nii.gz/h5
        # print(file_format)
        # if file_format == ".npy":
        #     raw_img = np.load(self.data_dict[name]["image"])
        #     label_img = np.load(self.data_dict[name]["label"])
        # if file_format == '.nii.gz':
        # raw_img = io.imread(self.data_dict[name]["image"], plugin='simpleitk')
        # label_img = io.imread(self.data_dict[name]["label"], plugin='simpleitk')
        # elif file_format == ".h5":
        #     hf = h5py.File(self.data_dict[name]["path"], 'r+')
        #     raw_img = np.array(hf["image"])
        #     label_img = np.array(hf["label"])
        #     hf.close()

        assert raw_img.shape == label_img.shape

        # 数据处理相关
        start_points=random3dcrop.random_crop_start_point(raw_img.shape)#起点
        raw_img_crop=random3dcrop(np.array(raw_img, float), start_points=start_points)
        label_img_crop=random3dcrop(np.array(label_img, float), start_points=start_points)
        raw_img_crop=normalization(raw_img_crop)

        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        label_img_crop = np.expand_dims(label_img_crop, axis=0)
        # print(raw_img_crop.shape,label_img_crop.shape)#(1, 32, 128, 128) (1, 32, 128, 128) 没有第一维N，一张一张图片处理
        output = {"image": raw_img_crop, "label": label_img_crop}

        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)
        # print(type(output['image']))#<class 'torch.Tensor'>
        # print(windowMax,crop_size)#600 (32, 128, 128)
        return output

    def to_tensor(self, images):
        for item in images.keys():
            images[item]=from_numpy(images[item]).float()
        return images

    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}

        for item in image_tensors.keys():
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob),tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),\
                tio.RandomNoise(p=prob),\
                    tio.RandomMotion(p=prob),\
                        tio.RandomBiasField(p=prob),\
                            tio.RandomSpike(p=prob),\
                                tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])

        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data

        return image_tensors


class airway_dataset2(Dataset):
    # train_dataset_more_focus_on_airways_of_low_generation = airway_dataset(dataset_info_more_focus_on_airways_of_low_generation)
    def __init__(self, data_dict, num_of_samples=None):
        # each item of data_dict is {name:{"image": img path, "label": label path}}
        self.data_dict = data_dict
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            chosen_names = np.random.choice(
                np.array(list(data_dict)), num_of_samples, replace=False
            )
        else:  # 只执行这个
            chosen_names = np.array(list(data_dict))
        # print(num_of_samples is None)#True
        # print(list(data_dict))#生成一个list，只包括这个字典的keys
        # print(chosen_names)#将上面的list(data_dict转为<class 'numpy.ndarray'>

        self.name_list = chosen_names
        # print('len',len(self.name_list))#36290 27535
        self.para = {}
        self.set_para()

    def get_name_list(self):
        return self.name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # print('idx{}',idx)#idx{} 11979 看得出是随机生成的
        # 下面调用的get函数,返回的是一个字典{'image':<class 'torch.Tensor'>&shape=torch.Size([1, 32, 128, 128]),
        #                                'label':<class 'torch.Tensor'>&shape=torch.Size([1, 32, 128, 128]),}
        # 并且数据经过了标准化、裁剪等处理
        return self.get(
            idx,
            file_format=self.para["file_format"],
            crop_size=self.para["crop_size"],
            windowMin=self.para["windowMin"],
            windowMax=self.para["windowMax"],
            need_tensor_output=self.para["need_tensor_output"],
            need_transform=self.para["need_transform"],
        )

    # 全部默认
    def set_para(
        self,
        file_format=".nii.gz",
        crop_size=32,
        windowMin=-1000,
        windowMax=150,
        need_tensor_output=True,
        need_transform=True,
    ):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    # 看了下参数全默认,只有idx是从getitem迭代得到的
    def get(
        self,
        idx,
        file_format=".nii.gz",
        crop_size=32,
        windowMin=-1000,
        windowMax=150,
        need_tensor_output=True,
        need_transform=True,
    ):

        random3dcrop = Random3DCrop_np(crop_size)
        normalization = Normalization_np(windowMin, windowMax)

        name = self.name_list[idx]

        raw_img = io.imread(self.data_dict[name]["image"], plugin="simpleitk")
        label_img = io.imread(self.data_dict[name]["label"], plugin="simpleitk")
        # 参数可选npy/nii.gz/h5
        # print(file_format)
        # if file_format == ".npy":
        #     raw_img = np.load(self.data_dict[name]["image"])
        #     label_img = np.load(self.data_dict[name]["label"])
        # if file_format == '.nii.gz':
        # raw_img = io.imread(self.data_dict[name]["image"], plugin='simpleitk')
        # label_img = io.imread(self.data_dict[name]["label"], plugin='simpleitk')
        # elif file_format == ".h5":
        #     hf = h5py.File(self.data_dict[name]["path"], 'r+')
        #     raw_img = np.array(hf["image"])
        #     label_img = np.array(hf["label"])
        #     hf.close()

        assert raw_img.shape == label_img.shape

        # 数据处理相关
        start_points = random3dcrop.random_crop_start_point(raw_img.shape)  # 起点
        raw_img_crop = random3dcrop(np.array(raw_img, float), start_points=start_points)
        label_img_crop = random3dcrop(
            np.array(label_img, float), start_points=start_points
        )
        raw_img_crop = normalization(raw_img_crop)

        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        label_img_crop = np.expand_dims(label_img_crop, axis=0)
        # print(raw_img_crop.shape,label_img_crop.shape)#(1, 32, 128, 128) (1, 32, 128, 128) 没有第一维N，一张一张图片处理
        output = {"image": raw_img_crop, "label": label_img_crop}

        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)
        # print(type(output['image']))#<class 'torch.Tensor'>
        # print(windowMax,crop_size)#600 (32, 128, 128)
        output["idx"] = idx
        return output

    def to_tensor(self, images):
        for item in images.keys():
            if isinstance(images[item], np.ndarray):
                # 如果是 numpy 数组，直接转换为 tensor
                images[item] = from_numpy(images[item]).float()
            elif isinstance(images[item], torch.Tensor):
                # 如果已经是 tensor 类型，直接转换为 float 类型
                images[item] = images[item].float()
            elif isinstance(images[item], int):
                # 如果是 int 类型，先转换为 numpy 数组，再转换为 tensor
                images[item] = from_numpy(np.array(images[item])).float()
            else:
                # 如果是其他类型，打印警告信息
                print(
                    f"警告: {item} 类型不符合预期，跳过转换。类型是: {type(images[item])}"
                )
        return images

    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio = {}

        for item in image_tensors.keys():
            dict_imgs_tio[item] = tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose(
            [
                tio.RandomFlip(axes=int(np.random.randint(3, size=1)[0]), p=prob),
                tio.RandomAffine(p=prob),
            ]
        )
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose(
            [
                tio.RandomBlur(p=prob),
                tio.RandomNoise(p=prob),
                tio.RandomMotion(p=prob),
                tio.RandomBiasField(p=prob),
                tio.RandomSpike(p=prob),
                tio.RandomGhosting(p=prob),
            ]
        )
        transformed_subject_all_imgs["image"] = transform_val(
            transformed_subject_all_imgs["image"]
        )

        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data

        return image_tensors


if __name__=="__main__":
    import pickle
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)
    data_dict = load_obj("../dataset_info/train_test_set_dict_EXACT09_LIDC_IDRI_128_set_1_extended_more_big_10")
    dataset = airway_dataset(data_dict)
    output = dataset.get(0)
    for name in output.keys():
        print(name, output[name].shape)
    num_workers = 1
    dataset.set_para(file_format='.npy', crop_size=64, windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True)
    Dataset_loader = DataLoader(dataset, batch_size=3, shuffle=True, \
        num_workers=num_workers, pin_memory=False, persistent_workers=False)
    batch = next(iter(Dataset_loader))
    for name in batch.keys():
        print(name, batch[name].shape)
