import numpy as np
import torch
import os
import skimage.io as io
from tqdm import tqdm
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import row_norms
from sklearn.cluster import KMeans as SKLearnKMeans


# change from kmeans_pytorch https://github.com/subhadarship/kmeans_pytorch
def initialize(X, num_clusters, init=None, random_state=None):

    if init is None:
        num_samples = len(X)
        indices = np.random.choice(num_samples, num_clusters, replace=False)
        initial_state = X[indices]
    elif torch.is_tensor(init):
        if init.shape[0] != num_clusters or init.shape[1] != X.shape[1]:
            raise ValueError(
                "The shape of the custom tensor 'init' should be (num_clusters, num_features)."
            )
        initial_state = init.to(X.device)
    else:
        raise ValueError(
            "Invalid type for 'init'. Use 'str' for predefined methods or 'torch.tensor' for custom initialization."
        )
    print(initial_state.shape)
    return initial_state


def kmeans(
    X,
    num_clusters,
    distance="euclidean",
    tol=1e-4,
    init=None,
    max_iter=300,
    device=torch.device("cpu"),
    random_state=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param init: (str or torch.tensor) initialization method [options: 'k-means++', 'random', or custom tensor]
                 [default: 'k-means++']
    :param max_iter: (int) maximum number of iterations [default: 300]
    :param device: (torch.device) device [default: cpu]
    :param random_state: (int or RandomState) seed or RandomState for reproducibility [default: None]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f"running k-means on {device}..")

    if distance == "euclidean":
        pairwise_distance_function = pairwise_distance
    elif distance == "cosine":
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    initial_state = initialize(X, num_clusters, init=init, random_state=random_state)

    iteration = 0
    tqdm_meter = tqdm(desc="[running kmeans]")
    while True:
        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
        )

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f"{iteration}",
            center_shift=f"{center_shift ** 2:0.6f}",
            tol=f"{tol:0.6f}",
        )
        tqdm_meter.update()
        if center_shift**2 < tol or iteration >= max_iter:
            break

    return choice_cluster.cpu(), initial_state.cpu()


def pairwise_distance(data1, data2, device=torch.device("cpu")):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device("cpu")):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def analysis_cluster(label_path, raw_case_name_list=None, slice_index=None):
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
        label_img = io.imread(label_img_addr, plugin="simpleitk")
        pixels_num = get_label_airway_pixels(label_img, slice_index)
        pixels_num_list.append(pixels_num)

    # 使用numpy计算均值和标准差
    mean_value = np.mean(pixels_num_list)
    std_deviation = np.std(pixels_num_list)
    num_sample = len(label_img_list)
    return mean_value, std_deviation, num_sample  # 均值和标准差


def mini_batch_kmeans(
    X,
    num_clusters,
    distance="euclidean",
    tol=1e-4,
    init=None,
    max_iter=3000,
    batch_size=64,
    device=torch.device("cpu"),
    random_state=None,
):
    """
    perform mini-batch kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) Dienstance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param init: (str or torch.tensor) initialization method [options: 'k-means++', 'random', or custom tensor]
                 [default: 'k-means++']
    :param max_iter: (int) maximum number of iterations [default: 300]
    :param batch_size: (int) size of each mini-batch [default: 64]
    :param device: (torch.device) device [default: cpu]
    :param random_state: (int or RandomState) seed or RandomState for reproducibility [default: None]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    print(f"running mini-batch k-means on {device}..")

    if distance == "euclidean":
        pairwise_distance_function = pairwise_distance
    elif distance == "cosine":
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # initialize
    initial_state = initialize(X, num_clusters, init=init, random_state=random_state)

    iteration = 0
    tqdm_meter = tqdm(desc="[running mini-batch kmeans]")
    while True:
        # 分批处理数据
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])

            # 将当前小批量数据加载到设备
            X_batch = X[start_idx:end_idx].to(device)

            dis = pairwise_distance_function(X_batch, initial_state)

            choice_cluster = torch.argmin(dis, dim=1)

            initial_state_pre = initial_state.clone()

            for index in range(num_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

                selected = torch.index_select(X_batch, 0, selected)
                initial_state[index] = selected.mean(dim=0)

            center_shift = torch.sum(
                torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))
            )

            # increment iteration
            iteration += 1

            # update tqdm meter
            tqdm_meter.set_postfix(
                iteration=f"{iteration}",
                center_shift=f"{center_shift ** 2:0.6f}",
                tol=f"{tol:0.6f}",
            )
            tqdm_meter.update()

            if center_shift**2 < tol or iteration >= max_iter:
                break

        if center_shift**2 < tol or iteration >= max_iter:
            break

    return choice_cluster.cpu(), initial_state.cpu()
