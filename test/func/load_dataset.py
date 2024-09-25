# dataset load and transform
import numpy as np
import skimage.io as io
import SimpleITK as sitk
import math
import random
import h5py

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio

import matplotlib.pyplot as plt

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
            start_points = random_crop_start(img_3d.shape)
        
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


class airway_3d_dataset_org(Dataset):
    def __init__(self, data_dict):
        # each item of data_dict is {name:{"raw":raw img path, "label": label img path}}
        self.data_dict = data_dict
        self.name_list = np.array(list(data_dict))
    
    def get_batch(self, generate_from_given_names=[], file_format='.npy', number_of_batches_we_need=10, crop_size=32, windowMin=-1000, windowMax=150):
        
        random3dcrop=Random3DCrop_np(crop_size)
        normalization=Normalization_np(windowMin, windowMax)
        
        if generate_from_given_names==[]:
            file_idx=np.arange(len(self.name_list), dtype=np.int)
            generate_from_given_names=self.name_list[file_idx[np.random.randint(0, len(file_idx), number_of_batches_we_need)]]
        
        print("generate batch from "+str(generate_from_given_names), end="\r")
        
        unique_crop_file_name, unique_name_counts=np.unique(generate_from_given_names, return_counts=True)
        
        raw_batch=[]
        label_batch=[]
        
        for idx, name in enumerate(unique_crop_file_name):
            if file_format == ".npy":
                raw_img = np.load(self.data_dict[name]["image"])
                label_img = np.load(self.data_dict[name]["label"])
            elif file_format == '.nii.gz':
                raw_img = io.imread(self.data_dict[name]["image"], plugin='simpleitk')
                label_img = io.imread(self.data_dict[name]["label"], plugin='simpleitk')
            elif file_format == ".h5":
                hf = h5py.File(self.data_dict[name]["path"], 'r+')
                raw_img = np.array(hf["image"])
                label_img = np.array(hf["label"])
                hf.close()
            
            raw_img = np.array(raw_img, np.float)
            label_img = np.array(label_img, np.float)
            
            assert raw_img.shape == label_img.shape
            print("load image shape: "+str(raw_img.shape))
            
            counts=0
            while counts<unique_name_counts[idx]:
                counts=counts+1
                start_points=random3dcrop.random_crop_start_point(raw_img.shape)
                raw_img_crop=random3dcrop(raw_img, start_points=start_points)
                label_img_crop=random3dcrop(label_img, start_points=start_points)
                raw_img_crop=normalization(raw_img_crop)
                
                raw_batch.append(raw_img_crop)
                label_batch.append(label_img_crop)
        
        if isinstance(crop_size, int):
            crop_size_for_every_dimension=(crop_size, crop_size, crop_size)
        else:
            assert len(crop_size)==3
            crop_size_for_every_dimension=crop_size
        
        number_of_channels = 1
        
        raw_batch=np.array(raw_batch)
        raw_batch=raw_batch.reshape(number_of_batches_we_need,
                                    number_of_channels,
                                    crop_size_for_every_dimension[0],
                                    crop_size_for_every_dimension[1],
                                    crop_size_for_every_dimension[2])
        
        label_batch=np.array(label_batch)
        label_batch=label_batch.reshape(number_of_batches_we_need,
                                        number_of_channels,
                                        crop_size_for_every_dimension[0],
                                        crop_size_for_every_dimension[1],
                                        crop_size_for_every_dimension[2])
        
        return {"image": raw_batch, "label": label_batch}
    
    """
    def to_tensor(self, images):  # images: a dict, each item should be in numpy.array format
        raw_tensor=from_numpy(np.array(images['image'],dtype=np.float)).float()
        label_tensor=from_numpy(np.array(images['label'],dtype=np.float)).float()
        
        return {'image': raw_tensor, 'label': label_tensor}
        """
    def to_tensor(self, images, device = torch.device('cpu')):
        images_tensor={}
        for item in images.keys():
            images_tensor[item]=from_numpy(images[item]).float().to(device)
        return images_tensor
    """
    def transform_transpose(self, images, prob=0.5, transpose=None):
        images_transformed={}
        if prob>=np.random.rand() or transpose is not None:
            if transpose is None:
                transpose = np.array([2,3,4])
                np.random.shuffle(transpose)
            for item in images.keys():
                images_transformed[item] = np.transpose(images[item], (images[item].shape[0], images[item].shape[1], int(transpose[0]), int(transpose[1]), int(transpose[2])))
            return images_transformed, transpose
        else:
            return images, transpose
            """
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            assert type(image_tensors[item] == "torch.Tensor")
            image_tensors[item] = torch.reshape(image_tensors[item], (image_tensors[item].shape[0],
                                                image_tensors[item].shape[2],
                                                image_tensors[item].shape[3],
                                                image_tensors[item].shape[4]))
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob),tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),
            tio.RandomNoise(p=prob),tio.RandomMotion(p=prob),tio.RandomBiasField(p=prob),tio.RandomSpike(p=prob),tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        for item in subject_all_imgs.keys():
            temp_img = transformed_subject_all_imgs[item].data
            assert type(temp_img == "torch.Tensor")
            image_tensors[item] = torch.reshape(temp_img, (temp_img.shape[0],1,temp_img.shape[1],temp_img.shape[2],temp_img.shape[3]))
        
        return image_tensors
    
class airway_3d_dataset(Dataset):
    def __init__(self, data_dict, image_dictname="image", label_dictname="label", transforms=None):
        # each item of data_dict is {name:{"raw":raw img path, "label": label img path}}
        self.data_dict = data_dict
        self.name_list = np.array(list(data_dict.keys()))
        self.transforms = transforms
        self.image_dictname = image_dictname
        self.label_dictname = label_dictname
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, idx):
        sample = self.read_img(self.data_dict[self.name_list[idx]][self.image_dictname], self.data_dict[self.name_list[idx]][self.label_dictname])
        for transform in self.transforms:
            sample = transform(sample)
        sample = self.to_numpy_arr(sample)
        sample = self.to_tensor(sample)
        
        return sample
    
    def read_one_image(self, path):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        return reader.Execute()
    
    def read_img(self, image_path, label_path):
        # read image and label
        image = self.read_one_image(image_path)
         # cast image and label
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkInt16)
        image = castImageFilter.Execute(image)

        label = self.read_one_image(label_path)
        castImageFilter.SetOutputPixelType(sitk.sitkInt8)
        label = castImageFilter.Execute(label)

        return {'image':image, 'label':label}
    
    def to_numpy_arr(self, sample):
        image, label = sample['image'], sample['label']
        
        # convert sample to tf tensors
        image_np = sitk.GetArrayFromImage(sample['image'])
        label_np = sitk.GetArrayFromImage(sample['label'])

        image_np = np.asarray(image_np, np.float32)
        label_np = np.asarray(label_np, np.float32)
        
        return {'image': image_np, 'label': label_np}
    
    def to_tensor(self, sample):  # images: a dict, each item should be in numpy.array format
        image_tensor=from_numpy(np.array(sample['image'], dtype=np.float)).float()
        label_tensor=from_numpy(np.array(sample['label'], dtype=np.float)).float()
        
        return {'image': image_tensor, 'label': label_tensor}

class ManualNormalization(object):
    """
    Normalize an image by mapping intensity with given max and min window level
    """
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)
        intensityWindowingFilter.SetOutputMinimum(0)
        intensityWindowingFilter.SetWindowMaximum(self.windowMax);
        intensityWindowingFilter.SetWindowMinimum(self.windowMin);
        
        image = intensityWindowingFilter.Execute(image)
        
        return {'image': image, 'label': label}

class Resample(object):
    """
    Resample the volume in a sample to a given voxel size
    Args:
    voxel_size (float or tuple): Desired output size.
    If float, output volume is isotropic.
    If tuple, output voxel size is matched with voxel size
    Currently only support linear interpolation method
    """
    def __init__(self, voxel_size):
        self.name = 'Resample'
        assert isinstance(voxel_size, (float, tuple))
        if isinstance(voxel_size, float):
            self.voxel_size = (voxel_size, voxel_size, voxel_size)
        else:
            assert len(voxel_size) == 3
            self.voxel_size = voxel_size
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        old_spacing = image.GetSpacing()
        old_size = image.GetSize()

        new_spacing = self.voxel_size

        new_size = []
        for i in range(3):
            new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
        new_size = tuple(new_size)

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(2)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)

        # resample on image
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        # print("Resampling image...")
        image = resampler.Execute(image)

        # resample on segmentation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputOrigin(label.GetOrigin())
        resampler.SetOutputDirection(label.GetDirection())
        # print("Resampling segmentation...")
        label = resampler.Execute(label)

        return {'image': image, 'label': label}

class Padding(object):
    """
    Add padding to the image if size is smaller than patch size
    Args:
    output_size (tuple or int): Desired output size. If int, a cubic volume is formed
    """
    def __init__(self, output_size):
        self.name = 'Padding'

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

        assert all(i > 0 for i in list(self.output_size))
    
    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        output_size = self.output_size
        if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
            return sample
        else:
            output_size = list(output_size)
            if size_old[0] > output_size[0]:
                output_size[0] = size_old[0]
            if size_old[1] > output_size[1]:
                output_size[1] = size_old[1]
            if size_old[2] > output_size[2]:
                output_size[2] = size_old[2]
 
            output_size = tuple(output_size)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(image.GetSpacing())
            resampler.SetSize(output_size)

            # resample on image
            resampler.SetInterpolator(2)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            image = resampler.Execute(image)

            # resample on label
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetOutputOrigin(label.GetOrigin())
            resampler.SetOutputDirection(label.GetDirection())

            label = resampler.Execute(label)
            
            return {'image': image, 'label': label}

class RandomCrop(object):
    """
    Crop randomly the image in a sample. This is usually used for data augmentation.
    Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
    This transformation only applicable in train mode
    Args:
    output_size (tuple or int): Desired output size. If int, cubic crop is made.
    """

    def __init__(self, output_size, drop_ratio=0.1, min_pixel=1):
        self.name = 'Random Crop'
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size
        
        assert isinstance(drop_ratio, float)
        if drop_ratio >=0 and drop_ratio<=1:
            self.drop_ratio = drop_ratio
        else:
            raise RuntimeError('Drop ratio should be between 0 and 1')
        
        assert isinstance(min_pixel, int)
        if min_pixel >=0 :
            self.min_pixel = min_pixel
        else:
            raise RuntimeError('Min label pixel count should be integer larger than 0')
    
    def __call__(self,sample):
        image, label = sample['image'], sample['label']
        size_old = image.GetSize()
        size_new = self.output_size
        
        contain_label = False

        roiFilter = sitk.RegionOfInterestImageFilter()
        roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

        while not contain_label: 
            # get the start crop coordinate in ijk
            if size_old[0] <= size_new[0]:
                start_i = 0
            else:
                start_i = np.random.randint(0, size_old[0]-size_new[0])
            if size_old[1] <= size_new[1]:
                start_j = 0
            else:
                start_j = np.random.randint(0, size_old[1]-size_new[1])
            
            if size_old[2] <= size_new[2]:
                start_k = 0
            else:
                start_k = np.random.randint(0, size_old[2]-size_new[2])
                
            roiFilter.SetIndex([start_i,start_j,start_k])
            label_crop = roiFilter.Execute(label)
            statFilter = sitk.StatisticsImageFilter()
            statFilter.Execute(label_crop)

            # will iterate until a sub volume containing label is extracted
            # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
            # if statFilter.GetSum()/pixel_count<self.min_ratio:
            if statFilter.GetSum()<self.min_pixel:
                contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
            else:
                contain_label = True
        
        image_crop = roiFilter.Execute(image)
        return {'image': image_crop, 'label': label_crop}
    
    def drop(self,probability):
        return random.random() <= probability

class RandomNoise(object):
    """
    Randomly noise to the image in a sample. This is usually used for data augmentation.
    """
    def __init__(self):
        self.name = 'Random Noise'
    
    def __call__(self, sample):
        self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
        self.noiseFilter.SetMean(0)
        self.noiseFilter.SetStandardDeviation(0.1)
        
        image, label = sample['image'], sample['label']
        image = self.noiseFilter.Execute(image)
        
        return {'image': image, 'label': label}

    
def show_samples(images):
    raw_img=images['raw'] # should be in numpy.array format
    label_img=images['label']
        
    shape_img=raw_img.shape
        
    assert len(shape_img)>=3
    assert len(shape_img)<=5
        
    if len(shape_img)==5:
        k=np.random.randint(0, shape_img[0]-1)
        raw_img=raw_img[k, 0]
        label_img=label_img[k, 0]
    elif len(shape_img)==4:
        raw_img=raw_img[0]
        label_img=label_img[0]
    
    shape_img=raw_img.shape
    N=np.random.randint(0, shape_img[2])
        
    plt.figure(figsize=(15, 8))
        
    plt.subplot(1, 2, 1)
    plt.title('Raw image')
    #plt.axis('off')
    plt.imshow(raw_img[N, :, :])
        
    plt.subplot(1, 2, 2)
    plt.title('Raw image with label')
    #plt.axis('off')
    plt.imshow(raw_img[N, :, :])
    plt.contour(label_img[N, :, :], levels=1, colors='r', linestyles='-', alpha=0.5)