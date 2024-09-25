import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk
import torch

#当我们需要对一个3D图像进行处理时，可能需要将其分成多个小块，以便于对每个小块进行处理，比如使用卷积神经网络进行分类或分割等任务。
# 这时，我们需要一个函数来将3D图像分成多个小块，这个函数就是crop_one_3d_img。
def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array
    # input_img：输入的3D图像，为numpy.array类型。
    # crop_cube_size：可以是一个int类型的值，也可以是一个长度为3的tuple类型的值，表示裁剪出来的小方块在3个方向上的大小
    # stride：stride也可以是一个int类型的值，也可以是一个长度为3的tuple类型的值，表示在3个方向上移动的步长
    assert isinstance(crop_cube_size, (int, tuple))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3#如果 crop_cube_size 是一个三元组，则检查其长度是否为 3
    
    #不能超过input_img的shape
    crop_cube_size = (min(crop_cube_size[0], input_img.shape[0]),
                      min(crop_cube_size[1], input_img.shape[1]),
                      min(crop_cube_size[2], input_img.shape[2]))
    
    #检查 stride 是否为整数或 3 元组，类似于 crop_cube_size。如果它是一个整数，则将其转换为一个大小为 3 的 numpy 数组
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3

    #获取输入图像的形状并计算需要切割的总次数
    img_shape=input_img.shape
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    
    count=0
    
    crop_list = []
    #现在，我们将开始对输入图像进行切割。
    # 通过使用三个 for 循环，我们遍历整个输入图像，每个循环中获取一个立方体。
    # 在每个迭代中，我们还检查当前位置是否可行，并相应地调整 x、y 和 z 的开始和结束索引。
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                # 接下来的代码是对于每个 i, j, k 的组合，判断裁剪的范围是否超出了原始图像的边界。
                # 如果没有超出边界，就根据裁剪范围和步长计算出在裁剪后的图像中的范围；如果超出了边界，则在原始图像的边界处进行裁剪。
                print('crop one 3d img progress : '+str(int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start_input=i
                    x_end_input=i+crop_cube_size[0]
                    x_start_output=i#只使用了x_start_input没有用x_start_output
                    x_end_output=i+stride[0]#同
                else:
                    x_start_input=img_shape[0]-crop_cube_size[0]
                    x_end_input=img_shape[0]
                    x_start_output=i
                    x_end_output=img_shape[0]

                #这部分代码的作用是计算y轴方向上需要裁剪的区域的起始和结束位置。
                # 如果当前位置j+crop_cube_size[1]小于等于img_shape[1]，说明可以完整地取出crop_cube_size[1]大小的区域，
                # 因此y_start_input从j开始，y_end_input从j+crop_cube_size[1]开始，y_start_output和y_end_output也相同。
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start_input=j
                    y_end_input=j+crop_cube_size[1]
                    y_start_output=j
                    y_end_output=j+stride[1]
                else:
                # 如果当前位置j+crop_cube_size[1]大于img_shape[1]，说明无法完整地取出crop_cube_size[1]大小的区域
                # 因此y_start_input为img_shape[1]-crop_cube_size[1]，y_end_input为img_shape[1]，
                # 表示从img_shape[1]-crop_cube_size[1]位置开始取到img_shape[1]位置，
                # 此时y_start_output为j，表示当前位置j之前的部分已经被裁剪过了，
                # y_end_output为img_shape[1]，表示当前位置j到img_shape[1]位置之间的部分会被裁剪。
                    y_start_input=img_shape[1]-crop_cube_size[1]
                    y_end_input=img_shape[1]
                    y_start_output=j
                    y_end_output=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start_input=k
                    z_end_input=k+crop_cube_size[2]
                    z_start_output=k
                    z_end_output=k+stride[2]
                else:
                    z_start_input=img_shape[2]-crop_cube_size[2]
                    z_end_input=img_shape[2]
                    z_start_output=k
                    z_end_output=img_shape[2]
                #最后，我们将裁剪后的图像添加到 crop_list 中，并递增计数器 count。循环结束后，我们将 crop_list 返回。
                crop_temp=input_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                crop_list.append(np.array(crop_temp, dtype=float))
                
                count=count+1
                
    return crop_list

import os
import numpy as np
import skimage.io as ioc  
import SimpleITK as sitk


# build the raw_data_dict for train

raw_data_dict = dict()

# LIDC-IDRI data
LIDC_IDRI_file_path = "/mnt/wangc/LIDC"
LIDC_IDRI_raw_path = LIDC_IDRI_file_path+"/image"
LIDC_IDRI_label_path = LIDC_IDRI_file_path+"/label"

LIDC_IDRI_raw_names = os.listdir(LIDC_IDRI_raw_path)
LIDC_IDRI_raw_names.sort()

LIDC_IDRI_label_names = os.listdir(LIDC_IDRI_label_path)
LIDC_IDRI_label_names.sort()

case_names = []

for case in LIDC_IDRI_raw_names:
    temp = case.split(".")[0]
    #print(temp)
    case_names.append(temp)
    raw_data_dict["LIDC_IDRI_"+temp]={}
    raw_data_dict["LIDC_IDRI_"+temp]["image"]=LIDC_IDRI_raw_path+"/"+case

for case in LIDC_IDRI_label_names:
    temp = case.split(".")[0]
    #print(temp)
    if temp in case_names:
        raw_data_dict["LIDC_IDRI_"+temp]["label"]=LIDC_IDRI_label_path+"/"+case

LIDC_IDRI_data_dict = raw_data_dict



crop_cube_size=(256, 256, 256)
stride=(128,128,128)

# -----INPUT-----
output_file_path = "/mnt/wangc/LIDC/Precrop_dataset_for_LIDC-IDRI"

if not os.path.exists(output_file_path+"/image/"):
    os.makedirs(output_file_path+"/image/")

if not os.path.exists(output_file_path+"/label/"):
    os.makedirs(output_file_path+"/label/")

raw_data_dict = LIDC_IDRI_data_dict
# -----END-----

for i, case in enumerate(raw_data_dict.keys()):
    if i<5:
        print(f'完成{i*100/40}% ')
    else:
        raw_img = io.imread(raw_data_dict[case]["image"], plugin='simpleitk')
        label_img = io.imread(raw_data_dict[case]["label"], plugin='simpleitk')
        
        raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
        label_img_crop_list = crop_one_3d_img(label_img, crop_cube_size=crop_cube_size, stride=stride)
        
        assert len(raw_img_crop_list)==len(label_img_crop_list)
        
        for idx in range(len(raw_img_crop_list)):
            print("progress: "+str(idx)+"th crop | "+str(i)+"th 3d img: "+str(case), end="\r")
            
            sitk.WriteImage(sitk.GetImageFromArray(raw_img_crop_list[idx]), output_file_path+"/image/"+case+"_"+str(idx)+".nii.gz")
            sitk.WriteImage(sitk.GetImageFromArray(label_img_crop_list[idx]), output_file_path+"/label/"+case+"_"+str(idx)+".nii.gz")
            
            # np.save(output_file_path+"/image/"+case+"_"+str(idx)+".npy", raw_img_crop_list[idx])
            # np.save(output_file_path+"/label/"+case+"_"+str(idx)+".npy", label_img_crop_list[idx])
        print(f'完成{i*100/40}% ')