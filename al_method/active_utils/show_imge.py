
import matplotlib.pyplot as plt
import os
import skimage.io as io
from torch import from_numpy as from_numpy
from matplotlib import gridspec

"""
读取raw_ima_path的文件
"""
#使用示例 show_all_2d_img_with_labels(raw_img_folder,slice_folder,img_num=2000,label_path=label_path)
def show_all_2d_img_with_labels(
    output_folder,      
    raw_img_path = None,  # 图像路径，raw_img_path 和 raw_img_list 二选一，没有则为None,都有则以 raw_img_path为准
    raw_img_list = None,  
    label_path=None, # 标签路径，当为None时，输出的图像不使用label
    img_num=None, # 显示的图像数量，当设置为None时，显示所有图像 
    num_images_per_batch=16, # 这个参数没调整好，请以默认值16位准
    slice_index=20, 
    file_name='no_name',
):
    if raw_img_path is None and raw_img_list is None:
        raise ValueError("Both raw_img_path and raw_img_list cannot be None. Please provide one.")
    elif raw_img_list is None:
        raw_img_list = os.listdir(raw_img_path)
    
    if img_num is None:
        img_num = len(raw_img_list)
    img_num = min(img_num, len(raw_img_list))
    
    # 检查 输出路径是否存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    num_batches = (img_num + num_images_per_batch - 1) // num_images_per_batch  # 将图像分成num_images_per_batch份
    num_rows = 4
    num_cols = 4
    for batch_num in range(num_batches):
        start_index = batch_num * num_images_per_batch
        end_index = min((batch_num + 1) * num_images_per_batch, img_num)
        
        img_list = []
        label_list = []  

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
