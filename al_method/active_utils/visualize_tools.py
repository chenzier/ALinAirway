from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from torch import from_numpy as from_numpy
from matplotlib import gridspec
import skimage.io as io
import os

"""
    读取raw_ima_path的文件
"""


# 使用示例 show_all_2d_img_with_labels(raw_img_folder,slice_folder,img_num=2000,label_path=label_path)
def show_all_2d_img_with_labels(
    output_folder,
    raw_img_path=None,  # 图像路径，raw_img_path 和 raw_img_list 二选一，没有则为None,都有则以 raw_img_path为准
    raw_img_list=None,
    label_path=None,  # 标签路径，当为None时，输出的图像不使用label
    img_num=None,  # 显示的图像数量，当设置为None时，显示所有图像
    num_images_per_batch=16,  # 这个参数没调整好，请以默认值16位准
    slice_index=20,
    file_name="no_name",
):
    if raw_img_path is None and raw_img_list is None:
        raise ValueError(
            "Both raw_img_path and raw_img_list cannot be None. Please provide one."
        )
    elif raw_img_list is None:
        raw_img_list = os.listdir(raw_img_path)

    if img_num is None:
        img_num = len(raw_img_list)
    img_num = min(img_num, len(raw_img_list))

    # 检查 输出路径是否存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_batches = (
        img_num + num_images_per_batch - 1
    ) // num_images_per_batch  # 将图像分成num_images_per_batch份
    num_rows = 4
    num_cols = 4
    for batch_num in range(num_batches):
        start_index = batch_num * num_images_per_batch
        end_index = min((batch_num + 1) * num_images_per_batch, img_num)

        img_list = []
        label_list = []

        img_names = []
        label_names = []

        for i in range(start_index, end_index):
            raw_img_addr = os.path.join(raw_img_path, raw_img_list[i])
            raw_img = io.imread(raw_img_addr, plugin="simpleitk")
            img_list.append(raw_img)

            img_names.append(raw_img_list[i])

            if label_path is not None:
                if raw_img_list is None:
                    label_img_list = os.listdir(label_path)
                else:
                    label_img_list = raw_img_list
                label_img_addr = os.path.join(
                    label_path, label_img_list[i]
                )  # 使用相同的索引加载标签图像
                label_img = io.imread(label_img_addr, plugin="simpleitk")
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
                            p = 0
                            while (
                                p < label_img.shape[0] and 1 not in label_img[p, :, :]
                            ):
                                p += 1
                            if p < label_img.shape[0]:
                                ax.imshow(raw_img[p, :, :], cmap="gray")
                                ax.contour(
                                    label_img[p, :, :], colors="r", linestyles="-"
                                )
                            else:
                                p = 0
                                ax.imshow(raw_img[p, :, :], cmap="gray")
                                ax.contour(
                                    label_img[p, :, :], colors="r", linestyles="-"
                                )
                                ax.text(
                                    1,
                                    1,
                                    "No Label",
                                    color="red",
                                    fontsize=16,
                                    ha="right",
                                    va="top",
                                    transform=ax.transAxes,
                                )
                        else:
                            p = slice_index
                            ax.imshow(raw_img[p, :, :], cmap="gray")
                            ax.contour(label_img[p, :, :], colors="r", linestyles="-")
                    else:
                        p = slice_index
                        ax.imshow(raw_img[p, :, :], cmap="gray")
                    ax.set_title(
                        f"{p} Image {img_names[index]}\nLabel {label_names[index]} "
                    )
                    ax.axis("off")

        # 调整子图之间的间距和布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(output_folder, f"{file_name}_{batch_num+1}.png"))

        # 关闭图像窗口，避免重叠
        plt.close()


# class_0_indices,class_0_names_filtere = visualize_and_return_indices(X_embedded_2d, cluster_labels, embeddings_dict,selected_indices=class_0_indices)
def visualize_and_return_indices(
    X_embedded_2d,
    cluster_labels,
    embeddings_dict,
    x1=100,
    y1=100,
    x2=None,
    y2=None,
    selected_indices=None,
    show_index=False,
    save_path=None,
):
    # 根据X_embedded_2d、cluster_labels、embeddings_dict将得到的聚类图绘出
    # 会去除一些值过于极端的点，具体和x1，x2，y1，y2有关
    # 此外还可以根据selected_indices，仅将需要选择点 在图像中绘出
    # show_index为True时，将对应点的索引也写出来

    # return值 filtered_indices表示这次选择了哪些点, names_filtered表示这些点的索引
    if x2 is None and y2 is None:
        filtered_indices = np.where(
            (abs(X_embedded_2d[:, 0]) < abs(x1)) & (abs(X_embedded_2d[:, 1]) < abs(y1))
        )[0]

    else:

        # 创建布尔索引
        x_condition = (X_embedded_2d[:, 0] >= x1) & (X_embedded_2d[:, 0] <= x2)
        y_condition = (X_embedded_2d[:, 1] >= y1) & (X_embedded_2d[:, 1] <= y2)  # 纵轴

        # 通过两个条件的逻辑与来筛选元素
        filtered_indices = np.where(x_condition & y_condition)[0]
    print("filtered_indices", len(filtered_indices))
    if selected_indices is not None:
        intersection_set = set(filtered_indices).intersection(selected_indices)
        filtered_indices = list(intersection_set)
        print("selected_indices", len(filtered_indices))
    # 筛选出符合阈值的数据
    filtered_embeddings = X_embedded_2d[filtered_indices]

    # 提取嵌入坐标和对应的名称
    names = list(embeddings_dict.keys())
    names_filtered = [names[i] for i in filtered_indices]
    unique_labels = np.unique(cluster_labels[filtered_indices])
    cmap = plt.get_cmap("viridis", len(unique_labels))
    colors = [cmap(i) for i in range(len(unique_labels))]
    custom_cmap = ListedColormap(colors)

    # 清除之前的图像，以防止重叠
    plt.clf()

    # 绘制图像
    plt.figure(figsize=(8, 6), dpi=100)
    if show_index is True:
        for i, j in enumerate(filtered_indices):
            plt.annotate(
                j,
                (filtered_embeddings[i, 0], filtered_embeddings[i, 1]),
                fontsize=8,
                alpha=0.7,
            )
        # print(i)

    num_classes = len(unique_labels)
    random_colors = np.random.rand(num_classes, 3)
    custom_cmap = ListedColormap(random_colors)

    plt.scatter(
        filtered_embeddings[:, 0],
        filtered_embeddings[:, 1],
        c=cluster_labels[filtered_indices],
        cmap=custom_cmap,
    )
    print("filtered_embeddings.shape", filtered_embeddings.shape)
    plt.title("t-SNE 2D Embedding with Cluster Coloring (Perplexity=70)")
    # sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=cluster_labels.min(), vmax=cluster_labels.max()))
    # sm._A = []  # 设置一个空的数组

    # 保存图片
    if save_path is None:
        save_path = "visualization.png"
    plt.savefig(save_path)

    # 返回结果
    return filtered_indices, names_filtered
