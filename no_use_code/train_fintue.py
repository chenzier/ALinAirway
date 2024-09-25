# 必须在python 3.8下才能运行 多了和少了都不行！
# 没有使用low_gen和high_gen
# train

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
import gc
import torch.distributed as dist
import sys
sys.path.append('..')
from func.load_dataset import airway_dataset
# from func.unet_3d_basic import UNet3D_basic
from func.model_arch import SegAirwayModel
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights, dice_loss, dice_loss_power
from func.ulti import load_obj

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
save_path = '/home/wangc/now/NaviAirway/checkpoint_11/random_20.pkl'  # 10,20,30,50,100
path_dataset_info_org = "/mnt/wangc/NaviAirway/data_info_11/train_dataset_info_20"

need_resume = True
# load_path = "/mnt/share102/cs22-wangc/EXACT09/abc_checkpoint_sample_org_33.pkl" #'checkpoint/checkpoint.pkl'
load_path = ""
learning_rate = 1e-5
max_epoch = 50  # 50
freq_switch_of_train_mode_high_low_generation = 1
num_samples_of_each_epoch = 20000  # 2000
batch_size = 8  # 4
# batch_size = 4
train_file_format = '.npy'
crop_size = (32, 128, 128)
windowMin_CT_img_HU = -1000  # min of CT image HU value
windowMax_CT_img_HU = 600  # max of CT image HU value
model_save_freq = 1
num_workers = 4
# ----------

# init model
model = SegAirwayModel(in_channels=1, out_channels=2)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 检查是否有多个GPU可用
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
print(device)
model.to(device)
print(321)
# 是否加载checkpoint
if need_resume and os.path.exists(load_path):
    print("resume model from "+str(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset
# 这个类似于json文件，重点在下个part的dataset

dataset_info_org = load_obj(path_dataset_info_org)
train_dataset_org = airway_dataset(dataset_info_org)
train_dataset_org.set_para(file_format=train_file_format, crop_size=crop_size,
                           windowMin=windowMin_CT_img_HU, windowMax=windowMax_CT_img_HU, need_tensor_output=True, need_transform=True)


print('total epoch: '+str(max_epoch))
print('len(dataset_info_org)', len(dataset_info_org))
start_time = time.time()

for ith_epoch in range(0, max_epoch):  # 注意ith_epoch从0开始

    sampler_of_airways_org = RandomSampler(train_dataset_org,
                                           num_samples=min(num_samples_of_each_epoch, len(dataset_info_org)), replacement=True)
    dataset_loader = DataLoader(train_dataset_org,
                                batch_size=batch_size, sampler=sampler_of_airways_org, num_workers=num_workers,
                                pin_memory=True, persistent_workers=(num_workers > 1))

    # 有多少个batch 比如有100个样本，batch_size=5，len(dataset_loader)=20
    len_dataset_loader = len(dataset_loader)

    for ith_batch, batch in enumerate(dataset_loader):
        img_input = batch['image'].float().to(device)

        groundtruth_foreground = batch['label'].float().to(device)
        groundtruth_background = 1-groundtruth_foreground

        # #代码根据标签数据计算前景像素和背景像素的数量，
        # # 然后计算前景像素和背景像素在整个图像中的比例（fore_pix_per和back_pix_per）。
        fore_pix_num = torch.sum(groundtruth_foreground)
        back_pix_num = torch.sum(groundtruth_background)
        fore_pix_per = fore_pix_num/(fore_pix_num+back_pix_num)
        back_pix_per = back_pix_num/(fore_pix_num+back_pix_num)
        # if back_pix_per*100 > 99.99:
        #     continue

       # print(fore_pix_num.shape,fore_pix_per.shape)#torch.Size([]) torch.Size([]) 都是单值tensor

        # 根据前景像素和背景像素的比例，代码计算一个权重张量weights，用来平衡前景像素和背景像素在训练中的影响。
        # 具体来说，weights的值在前景像素和背景像素处分别为
        # torch.exp(back_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))
        # 和
        # torch.exp(fore_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))，
        # 这两个值分别表示前景像素和背景像素的权重。
        # 通过这个权重张量，可以在训练过程中平衡前景像素和背景像素的重要性，从而提高模型的训练效果。
        # torch.eq()对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False。
        weights = (torch.exp(back_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground, 1).float() +
                   torch.exp(fore_pix_per)/(torch.exp(fore_pix_per)+torch.exp(back_pix_per))*torch.eq(groundtruth_foreground, 0).float()).to(device)
        # print((weights.shape)) #torch.Size([4, 1, 32, 128, 128])
        # weights=torch.tensor(1.0)
        # weights=weights.to(device)
        # print(weights.device)
        # 代码首先根据输入图像数据（img_input）调用模型，得到模型的输出结果（img_output）。
        img_output = model(img_input)

        # 然后，代码根据输出结果和标签数据（groundtruth_foreground和groundtruth_background）计算一个loss，用来评估模型的训练效果。
        # 这里使用的是dice_loss_weights和dice_loss_power_weights两种损失函数。
        # dice_loss_weights是一种加权的Dice Loss，用来平衡前景像素和背景像素在训练中的影响。
        # dice_loss_power_weights是一种加权的Power Dice Loss，它不仅考虑前景像素和背景像素的权重，还考虑它们之间的距离。
        # 这两种损失函数的具体实现可以参考相应的论文和代码实现。
        # dice_loss_weights是Lbg,dice_loss_power_weights是Law
        # 为什么第二维长度为2，第二维是通道维，按理说应该是1？？？？？？？？？原文说一个对应airway，一个对应气道
        # loss=Law+Lbg，一个对应前景，一个对应背景
        loss = dice_loss_weights(img_output[:, 0, :, :, :], groundtruth_background, weights) +\
            dice_loss_power_weights(
                img_output[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2)
        # 基于dice loss,loss是一种新的细支气管敏感损失
        # 解决两个问题:1.骰子损失是什么？2.相较于骰子损失，改变在哪里？ 结合原文

        # 接下来，代码根据模型的输出结果和标签数据计算模型的精度（accuracy），用来评估模型的性能。
        # 这里使用的是Dice系数作为模型的精度指标。Dice系数是一种广泛用于评估二值图像分割的指标，
        # 它表示模型预测的前景像素与实际前景像素的重叠度。
        # Dice系数的取值范围是0到1，取值越大表示模型的性能越好。
        accuracy = dice_accuracy(
            img_output[:, 1, :, :, :], groundtruth_foreground)

        # 最后，代码根据损失函数计算的梯度调用optimizer.zero_grad()清空模型的梯度信息，
        # 然后调用loss.backward()计算模型的梯度，最后调用optimizer.step()更新模型的参数。
        # 这个过程被称为反向传播（backpropagation），它是训练深度学习模型的核心步骤之一。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_consumption = time.time() - start_time

        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}/{3}]\t"
            "time(sec) {time:.2f}\t"
            "loss {loss:.4f}\t"
            "acc {acc:.2f}%\t"
            "fore pix {fore_pix_percentage:.2f}%\t"
            "back pix {back_pix_percentage:.2f}%\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                len_dataset_loader,
                time=time_consumption,
                loss=loss.item(),
                acc=accuracy.item()*100,
                fore_pix_percentage=fore_pix_per*100,
                back_pix_percentage=back_pix_per*100))

        # 这段代码用于删除dataset_loader对象并释放内存，以避免内存泄漏。
        # 在Python中，使用del关键字可以删除对象，从而释放它占用的内存。
        # 同时，为了确保对象被完全删除，可以调用Python内置的gc.collect()函数，它会主动进行垃圾回收，释放所有未引用的对象所占用的内存。
        # 由于训练过程中会不断加载新的数据集并创建新的dataset_loader对象，
        # 因此及时删除旧的对象和释放内存是非常重要的，可以避免内存占用过高导致程序崩溃。
    del dataset_loader
    gc.collect()


# model_save_freq = 1
# 这段代码是用来保存模型的。在每model_save_freq个epoch结束时，将当前模型的参数保存到指定的文件路径中。
# 具体来说，当(ith_epoch+1)%model_save_freq==0时，表示已经训练完了model_save_freq个epoch，需要保存模型。
# 代码中首先打印出当前epoch的编号，然后将模型的参数保存到指定的文件路径（save_path）中。
# 在保存模型参数之前，先将模型移到CPU上（model.to(torch.device('cpu'))），
# 这是因为如果当前模型在GPU上，保存模型参数时需要先将参数从GPU中取出，再保存到文件中，这样会占用较多的显存。
# 将模型移到CPU上可以避免这个问题。保存完模型参数之后，再将模型移到GPU上（model.to(device)），以便继续进行后续的训练。
    if (ith_epoch+1) % model_save_freq == 0:
        print('epoch: '+str(ith_epoch+1)+' save model')
        model.to(torch.device('cpu'))
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        model.to(device)

# 将模型移到CPU上会导致模型的计算速度变慢，因此最好只在保存模型参数时将模型移到CPU上，然后在继续训练之前再将模型移到GPU上。
# 同时，如果模型的参数比较大，将模型移到CPU上也可能会导致程序的内存占用较高，因此需要根据实际情况进行调整。
# 将模型从GPU移到CPU，保存完了再移到GPU是为了避免GPU内存占用过多。
# 在训练模型时，模型的参数会保存在GPU显存中，
# 如果在GPU上保存模型参数，需要将参数从GPU中读取，然后保存到文件中，这个过程会占用大量的GPU显存。
# 而将模型移到CPU上，可以将模型参数从GPU显存中移动到CPU内存中，从而释放GPU显存，避免程序运行缓慢或者崩溃的问题。
# 在保存完模型参数之后，再将模型移到GPU上，可以继续进行后续的训练。


# 虽然将模型移到CPU上会花费一定的时间，但是这个时间与模型参数的大小和GPU显存的使用情况有关，
# 如果模型参数比较小，将模型移到CPU上的时间也比较短，同时可以避免GPU显存占用过多，从而提高程序的运行速度和稳定性。
# 因此，在某些情况下，将模型移到CPU上保存模型参数是一个比较好的选择。
