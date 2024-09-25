import pickle
import numpy as np
import pandas as pd
import skimage.io as io
import SimpleITK as sitk
import os

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array
    assert isinstance(crop_cube_size, (int, tuple))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3
    
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3
    
    img_shape=input_img.shape
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    
    count=0
    
    crop_list = []
    
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('crop one 3d img progress : '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start=i
                    x_end=i+crop_cube_size[0]
#                     x_start_output=i
#                     x_end_output=i+stride[0]
                else:
                    x_start=img_shape[0]-crop_cube_size[0]
                    x_end=img_shape[0]
#                     x_start_output=i
#                     x_end_output=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start=j
                    y_end=j+crop_cube_size[1]
#                     y_start_output=j
#                     y_end_output=j+stride[1]
                else:
                    y_start=img_shape[1]-crop_cube_size[1]
                    y_end=img_shape[1]
#                     y_start_output=j
#                     y_end_output=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start=k
                    z_end=k+crop_cube_size[2]
#                     z_start_output=k
#                     z_end_output=k+stride[2]
                else:
                    z_start=img_shape[2]-crop_cube_size[2]
                    z_end=img_shape[2]
#                     z_start_output=k
#                     z_end_output=img_shape[2]
                
                crop_temp=input_img[x_start:x_end, y_start:y_end, z_start:z_end]
                crop_list.append(np.array(crop_temp, dtype=np.float))
                
                count=count+1
                
    return crop_list

def load_one_CT_img(img_path):
    return io.imread(img_path, plugin='simpleitk')

def load_many_CT_img(img_path):
    img_dict = {}
    for filename in os.listdir(img_path):
        img_dict[filename[:6]]=load_one_CT_img(os.path.join(img_path, filename))
    return img_dict

def loadFile(filename):
    # 函数loadFile(filename)用于读取指定路径下的单张DICOM格式的图像，
    ds = sitk.ReadImage(filename)#读取图像
    #pydicom.dcmread(filename)# 需要注意的是，函数中也出现了注释掉的另一种读取DICOM格式图像的方式（使用pydicom库），但这种方式并没有被使用。
    img_array = sitk.GetArrayFromImage(ds)#将读取的图像转换成像素数组。
    frame_num, width, height = img_array.shape#获得该图像的高、宽、帧数信息(frame_num, width, height)
    #print("frame_num, width, height: "+str((frame_num, width, height)))
    return img_array, frame_num, width, height# 返回该图像的像素数组img_array，以及该图像的高、宽、帧数信息(frame_num, width, height)。
# frame_num是指DICOM图像中的帧数，也就是指图像中有多少张图片。
# DICOM图像可以包含多帧，每一帧都是一张图片。因此，frame_num表示的是DICOM图像中有多少个这样的帧。
# 在函数loadFile(filename)中，通过读取DICOM图像中的像素数组，然后查看该数组的形状，就可以得到帧数信息(frame_num)。
# 如果帧数等于1，表示该DICOM图像只包含一帧，否则就是多帧。
# 需要注意的是，函数loadFile(filename)只会读取DICOM图像的第一帧，因为它只返回了一张图片的像素数组。



def get_3d_img_for_one_case(img_path_list, img_format="dcm"):
    # 用于将指定路径下的 所有图像 按照顺序 合成 为一个三维图像，
    # 需要注意的是，函数中的img_format参数同样没有被使用，因此该参数可以忽略。
    img_3d=[]
    for idx, img_path in enumerate(img_path_list):
        print("progress: "+str(idx/len(img_path_list))+"; "+str(img_path), end="\r")
        img_slice, frame_num, _, _ = loadFile(img_path)#
        assert frame_num==1
        img_3d.append(img_slice)
    img_3d=np.array(img_3d)#将img_3d转换为numpy数组，然后使用reshape()函数将其重新排列成三维图像的形状。返回合成的三维图像。
    return img_3d.reshape(img_3d.shape[0], img_3d.shape[2], img_3d.shape[3])


def get_and_save_3d_img_for_one_case(img_path, output_file_path, img_format="dcm"):
    case_names=os.listdir(img_path)
    case_names.sort()
    img_path_list = []
    for case_name in case_names:
        img_path_list.append(img_path+"/"+case_name)
    img_3d = get_3d_img_for_one_case(img_path_list)
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),output_file_path)#将img_3d保存到指定的文件路径(output_file_path)中。
    
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
    
def get_CT_image(image_path, windowMin=-1000, windowMax=600, need_norm=True):
    raw_img = io.imread(image_path, plugin='simpleitk')
    raw_img = np.array(raw_img, dtype=np.float)
    
    if need_norm:
        normalization=Normalization_np(windowMin=windowMin, windowMax=windowMax)
        return normalization(raw_img)
    else:
        return raw_img

# show the airway centerline
def get_df_of_centerline(connection_dict):
    d = {}
    d["x"] = []
    d["y"] = []
    d["z"] = []
    d["val"] = []
    d["text"] = []
    for item in connection_dict.keys():
        print(item, end="\r")
        d["x"].append(connection_dict[item]['loc'][0])
        d["y"].append(connection_dict[item]['loc'][1])
        d["z"].append(connection_dict[item]['loc'][2])
        d["val"].append(connection_dict[item]['generation'])
        d["text"].append(str(item)+": "+str({"before":connection_dict[item]["before"], "next":connection_dict[item]["next"]}))
    df = pd.DataFrame(data=d)
    return df

# show the airway centerline
def get_df_of_line_of_centerline(connection_dict):
    d = {}
    for label in connection_dict.keys():
        if connection_dict[label]["before"][0]==0:
            start_label = label
            break
    def get_next_point(connection_dict, current_label, d, idx):
        while (idx in d.keys()):
            idx+=1
        
        d[idx]={}#新建一个字典
        #将坐标、世代数存储到相应的列表中
        if "x" not in d[idx].keys():
            d[idx]["x"]=[]
        if "y" not in d[idx].keys():
            d[idx]["y"]=[]
        if "z" not in d[idx].keys():
            d[idx]["z"]=[]
        if "val" not in d[idx].keys():
            d[idx]["val"]=[]
        
        before_label = connection_dict[current_label]["before"][0]
        if before_label not in connection_dict.keys():
            before_label = current_label
        d[idx]["x"].append(connection_dict[before_label]["loc"][0])
        d[idx]["y"].append(connection_dict[before_label]["loc"][1])
        d[idx]["z"].append(connection_dict[before_label]["loc"][2])
        d[idx]["val"].append(connection_dict[before_label]["generation"])
        
        d[idx]["x"].append(connection_dict[current_label]["loc"][0])
        d[idx]["y"].append(connection_dict[current_label]["loc"][1])
        d[idx]["z"].append(connection_dict[current_label]["loc"][2])
        d[idx]["val"].append(connection_dict[current_label]["generation"])
        
        if connection_dict[current_label]["number_of_next"]==0:
            return
        else:
            for next_label in connection_dict[current_label]["next"]:
                get_next_point(connection_dict, next_label, d, idx+1)
    
    get_next_point(connection_dict, start_label, d,0)
#     生成的字典d包含了整个空气道中心线的坐标和对应的值信息。其中每个键值对的键为整数，代表了空气道中心线上的每一个点；
# 每个键值对的值为一个字典，包含了该点的三维坐标和对应的值。具体来说，键值对的值包括以下四个键：
                        # "x": 该点在x轴上的坐标值，类型为列表；
                        # "y": 该点在y轴上的坐标值，类型为列表；
                        # "z": 该点在z轴上的坐标值，类型为列表；
                        # "val": 该点的值，类型为列表。
# 其中，x、y、z分别是该点的三维坐标值，val则是该点对应的代数距离值，即在构建空气道中心线时被赋予的label值，代表该点与空气道根节点的距离。
    return d