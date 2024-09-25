import numpy as np
import torch
from torch import from_numpy as from_numpy
import os
import skimage.io as io

# how to run model
"""
raw_img, label = get_image_and_label(image_path, label_path)
raw_img, label = get_crop_of_image_and_label_within_the_range_of_airway_foreground(raw_img,label)
seg_result = semantic_segment_crop_and_cat(raw_img, model, device, crop_cube_size=[32, 128, 128], stride=[16, 64, 64])
seg_onehot = np.array(seg_result>threshold, dtype=np.int)
"""


def get_image_and_label(image_path, label_path):
    raw_img = io.imread(image_path, plugin='simpleitk')
    raw_img = np.array(raw_img, dtype=np.float)
    label = io.imread(label_path, plugin='simpleitk')
    label = np.array(label, dtype=np.float)

    return raw_img, label


def get_crop_of_image_and_label_within_the_range_of_airway_foreground(raw_img, label):
    locs = np.where(label > 0)
    x_min = np.min(locs[0])
    x_max = np.max(locs[0])
    y_min = np.min(locs[1])
    y_max = np.max(locs[1])
    z_min = np.min(locs[2])
    z_max = np.max(locs[2])

    return raw_img[x_min:x_max, y_min:y_max, z_min:z_max], label[x_min:x_max, y_min:y_max, z_min:z_max]


class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int, float))
        assert isinstance(windowMin, (int, float))
        self.windowMax = windowMax
        self.windowMin = windowMin

    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm -= np.min(img_3d_norm)
        max_99_val = np.percentile(img_3d_norm, 99)
        if max_99_val > 0:
            img_3d_norm = img_3d_norm/max_99_val*255

        return img_3d_norm

# seg_result_semi_supervise_learning = semantic_segment_crop_and_cat(raw_img,
#                                                                    model,
#                                                                    device,
#                                                                    crop_cube_size=[32, 128, 128],
#                                                                    stride=[16, 64, 64],
#                                                                    windowMin=-1000, windowMax=600)


def get_embedding(raw_patch,  # 图像
                  model,
                  device,  # 放在哪个设备上
                  crop_cube_size=256,  # 进行分割时每个子块的大小，可以是整数或长度为3的列表或元组
                  stride=256,  # 子块之间的间隔，可以是整数或长度为3的列表或元组
                  windowMin=-1000,
                  windowMax=600):
    raw_patch = from_numpy(raw_patch).float().to(device)  # 将图像转为张量

    with torch.no_grad():  # 关闭梯度计算
        embedding = model.get_embedding(raw_img_crop)  # 使用model得到预测mask
    return embedding


def semantic_segment_crop_and_cat(raw_img,  # 图像
                                  model,
                                  device,  # 放在哪个设备上
                                  crop_cube_size=256,  # 进行分割时每个子块的大小，可以是整数或长度为3的列表或元组
                                  stride=256,  # 子块之间的间隔，可以是整数或长度为3的列表或元组
                                  windowMin=-1000,
                                  windowMax=600):

    normalization = Normalization_np(
        windowMin, windowMax)  # 函数首先对 raw_img 应用了窗位窗宽归一化处理
    # 将像素值限制在 windowMin 和 windowMax 之间 同时保持图像的对比度和亮度

    raw_img = normalization(raw_img)

    # raw_img: 3d matrix, numpy.array
    # 将crop_cube_size和stride转为三元组
    assert isinstance(crop_cube_size, (int, list))
    if isinstance(crop_cube_size, int):
        crop_cube_size = np.array(
            [crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size) == 3

    assert isinstance(stride, (int, list))
    if isinstance(stride, int):
        stride = np.array([stride, stride, stride])
    else:
        assert len(stride) == 3
    # 调整stride和crop_cube_size，确保整个raw_img被覆盖
    for i in [0, 1, 2]:
        while crop_cube_size[i] > raw_img.shape[i]:
            crop_cube_size[i] = np.int32(crop_cube_size[i]/2)
            stride[i] = crop_cube_size[i]

    img_shape = raw_img.shape

    seg = np.zeros(img_shape)
    # 0 means this pixel has not been segmented, 1 means this pixel has been
    seg_log = np.zeros(img_shape)

    total = len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))\
        * len(np.arange(0, img_shape[2], stride[2]))
    # x 轴方向上 子块的个数：len(np.arange(0, img_shape[0], stride[0]))
    # y 轴方向上 子块的个数：len(np.arange(0, img_shape[1], stride[1]))
    # z 轴方向上 子块的个数：len(np.arange(0, img_shape[2], stride[2]))
    # 因此，通过将上述三个方向上的子块数相乘即可得到总的子块数。
    # 由于每个子块都需要进行语义分割操作，所以 total 变量的值表示了需要进行的总操作数。

    count = 0  # 目前已经处理完的切割次数

    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('Progress of segment_3d_img: ' +
                      str(np.int32(count/total*100))+'%', end='\r')

                # 分块 循环变量 i, j, k 分别表示当前块的起始位置。
                if i+crop_cube_size[0] <= img_shape[0]:
                    x_start_input = i
                    x_end_input = i+crop_cube_size[0]
                else:
                    x_start_input = img_shape[0]-crop_cube_size[0]
                    x_end_input = img_shape[0]

                if j+crop_cube_size[1] <= img_shape[1]:
                    y_start_input = j
                    y_end_input = j+crop_cube_size[1]
                else:
                    y_start_input = img_shape[1]-crop_cube_size[1]
                    y_end_input = img_shape[1]

                if k+crop_cube_size[2] <= img_shape[2]:
                    z_start_input = k
                    z_end_input = k+crop_cube_size[2]
                else:
                    z_start_input = img_shape[2]-crop_cube_size[2]
                    z_end_input = img_shape[2]

                # 截取raw_img、seg_log、seg_crop 的切片
                raw_img_crop = raw_img[x_start_input:x_end_input,
                                       y_start_input:y_end_input, z_start_input:z_end_input]
                raw_img_crop = normalization(raw_img_crop)

                seg_log_crop = seg_log[x_start_input:x_end_input,
                                       y_start_input:y_end_input, z_start_input:z_end_input]
                seg_crop = seg[x_start_input:x_end_input,
                               y_start_input:y_end_input, z_start_input:z_end_input]
                # 在Python中，try 和 except 通常被用于捕获代码执行过程中的异常情况，从而让程序更加健壮和容错。
                # try 块内包含可能会引发异常的语句，当这些语句发生异常时，就会触发 except 块中的代码来处理异常情况。
                try:
                    raw_img_crop = raw_img_crop.reshape(
                        1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])
                    # 在这段代码中，try 块中的语句是
                    # raw_img_crop=raw_img_crop.reshape(1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2])，
                    # 这个语句尝试将 raw_img_crop 数组转换成形状为 (1, 1, crop_cube_size[0], crop_cube_size[1], crop_cube_size[2]) 的张量。
                    # 如果 raw_img_crop 数组已经具有这个形状，那么这个语句不会有任何问题；
                    # 但如果数组的形状与指定的形状不匹配，就会引发异常。
                except:
                    # 如果发生异常，程序就会跳转到 except 块中。在这里，打印了一些有用的调试信息，以便找到问题所在
                    print("raw_img_crop shape: "+str(raw_img_crop.shape))
                    print("raw_img shape: "+str(raw_img.shape))
                    print("i, j, k: "+str((i, j, k)))
                    print("crop from: "+str((x_start_input, x_end_input,
                          y_start_input, y_end_input, z_start_input, z_end_input)))
                raw_img_crop = from_numpy(
                    raw_img_crop).float().to(device)  # 将图像转为张量

                with torch.no_grad():  # 关闭梯度计算
                    seg_crop_output = model(raw_img_crop)  # 使用model得到预测mask
                    # print(seg_crop_output.shape)#torch.Size([1, 2, 32, 128, 128])
                    # 注意这个结果通道维为2，即第二个维度长度为2，seg_crop_output[:,0,:]是对前景的预测，seg_crop_output[:,1,:]是对背景的预测
                seg_crop_output_np = seg_crop_output[:, 1, :, :, :].cpu(
                ).detach().numpy()  # 转为numpy
                # print('aa',seg_crop_output_np.shape)#(1,32,128,128)
                # 降维,shape有如下变化
                seg_crop_output_np = seg_crop_output_np[0, :, :, :]
                # print(seg_crop_output_np.shape)#(32,128,128)
                seg_temp = np.zeros(seg_crop.shape)
                # 对于 seg_log_crop 中像素值为 1 的像素，
                # 将 seg_temp 中对应像素的值设置为 seg_crop_output_np 和 seg_crop 中对应像素值的平均值，

                # 循环第一次，seg_log_crop为全0，seg_temp为全0，seg_crop_output_np为模型预测结果的切片，所以第一次，seg_tempseg_crop_output_np
                # 第二次开始，seg_temp就不一定是全0，因为保存了第一次循环的结果，所以为了更精确的计算，会取两次结果(新的预测+前一次预测)的平均值
                # np.multiply(seg_crop_output_np[seg_log_crop==1],seg_crop[seg_log_crop==1])
                seg_temp[seg_log_crop == 1] = (
                    seg_crop_output_np[seg_log_crop == 1]+seg_crop[seg_log_crop == 1])/2
                # 对于 seg_log_crop 中像素值为 0 的像素，
                # 则将 seg_temp 中对应像素的值设置为 seg_crop_output_np 中对应像素值。
                seg_temp[seg_log_crop ==
                         0] = seg_crop_output_np[seg_log_crop == 0]
                # 最后，将 seg_temp 复制回 seg 中，同时将 seg_log 对应位置的像素值设置为 1
                # 将 count 加一，表示已经处理了一个分割块。最终返回 seg。
                seg[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input] =\
                    seg_temp
                seg_log[x_start_input:x_end_input, y_start_input:y_end_input,
                        z_start_input:z_end_input] = 1  # 用来记录哪一部分已经被处理了

                count = count+1

    return seg


def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.flatten()
    tflat = target.flatten()

    intersection = 2. * np.sum(np.multiply(iflat, tflat))

    A_sum = np.sum(np.multiply(iflat, iflat))
    B_sum = np.sum(np.multiply(tflat, tflat))

    return (intersection) / (A_sum + B_sum + 0.0001)
