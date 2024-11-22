import numpy as np
import skimage.io as io
from torch import from_numpy as from_numpy
import pickle


"""
    该函数用于获取嵌入表示（embedding）列表和字典。
    参数说明：
        Precrop_dataset_path: 预裁剪数据集的路径，用于定位图像和标签文件所在的目录
        raw_case_name_list: 原始病例名称列表，包含了要处理的图像对应的名称
        N: 要处理的图像数量上限，用于控制循环处理的次数
        model: 已经初始化好的模型对象，用于获取图像的嵌入表示（embedding）
        device: 用于指定模型运行的设备，例如 'cpu' 或 'cuda' 等
        random3dcrop: 用于进行随机三维裁剪的对象，应该具有相应的裁剪方法
        normalization: 用于对图像进行归一化处理的函数对象
        only_positive: 布尔类型参数，如果设置为True，则只处理图像中包含正例（例如标签中有特定值表示正例）的情况，默认为False
        need_embedding: 整数类型参数，用于指定要获取的嵌入表示的索引，如果不在有效范围内则使用模型的默认输出，默认为3
"""


def get_embeddings(
    Precrop_dataset_path,
    raw_case_name_list,
    N,
    model,
    device,
    random3dcrop,
    normalization,
    only_positive=False,  # 如果仅需要正例请选择
    need_embedding=3,
):
    i = 0
    embeddings_list = []  # 用于存储每个图像的 embeddings[3]
    embeddings_dict = {}

    for name in raw_case_name_list:
        if i < N:
            i += 1
            img_addr = Precrop_dataset_path + "/image" + "/" + name
            print(f"this is {i}")  # todo:换成log
            img = io.imread(img_addr, plugin="simpleitk")

            start_points = random3dcrop.random_crop_start_point(img.shape)  # 起点
            raw_img_crop = random3dcrop(np.array(img, float), start_points=start_points)
            raw_img_crop = normalization(raw_img_crop)
            raw_img_crop = np.expand_dims(raw_img_crop, axis=0)

            if only_positive is True:
                label_addr = Precrop_dataset_path + "/label" + "/" + name
                label_img = io.imread(label_addr, plugin="simpleitk")
                if 1 in label_img:
                    img_tensor = from_numpy(raw_img_crop).float()
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    embeddings = model.get_embedding(img_tensor)
                    if 0 <= need_embedding < len(embeddings):
                        emb = embeddings[need_embedding].cpu().detach().numpy()
                    else:
                        emb = model(img_tensor).cpu().detach().numpy()
                    embeddings_list.append(emb)  # 移动到CPU并转为NumPy后添加到列表
                    embeddings_dict[name] = emb
                    del img_tensor
            else:
                img_tensor = from_numpy(raw_img_crop).float()
                img_tensor = img_tensor.unsqueeze(0).to(device)
                embeddings = model.get_embedding(img_tensor)
                if 0 <= need_embedding < len(embeddings):
                    emb = embeddings[need_embedding].cpu().detach().numpy()
                else:
                    emb = model(img_tensor).cpu().detach().numpy()
                embeddings_list.append(emb)  # 移动到CPU并转为NumPy后添加到列表
                embeddings_dict[name] = emb
                del img_tensor

    return embeddings_list, embeddings_dict


def load_partial_embeddings(file_path1, file_path2, train_names=None, test_names=None):
    with open(file_path1, "rb") as file:
        loaded_data = pickle.load(file)
        exact_embeddings_list = loaded_data["embeddings_list"]
        exact_embeddings_dict = loaded_data["embeddings_dict"]

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)

    with open(file_path2, "rb") as file:
        loaded_data = pickle.load(file)
        lidc_embeddings_list = loaded_data["embeddings_list"]
        lidc_embeddings_dict = loaded_data["embeddings_dict"]
    # print('exact',len(exact_embeddings_dict),'lidc',len(lidc_embeddings_dict))
    lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)
    names = [
        "EXACT09_CASE01",
        "EXACT09_CASE02",
        "EXACT09_CASE03",
        "EXACT09_CASE04",
        "EXACT09_CASE05",
        "EXACT09_CASE06",
        "EXACT09_CASE07",
        "EXACT09_CASE08",
        "EXACT09_CASE09",
        "EXACT09_CASE10",
        "EXACT09_CASE11",
        "EXACT09_CASE12",
        "EXACT09_CASE13",
        "EXACT09_CASE14",
        "EXACT09_CASE15",
        "EXACT09_CASE16",
        "EXACT09_CASE17",
        "EXACT09_CASE18",
        "EXACT09_CASE19",
        "EXACT09_CASE20",
        "LIDC_IDRI_0066",
        "LIDC_IDRI_0140",
        "LIDC_IDRI_0328",
        "LIDC_IDRI_0376",
        "LIDC_IDRI_0403",
        "LIDC_IDRI_0430",
        "LIDC_IDRI_0438",
        "LIDC_IDRI_0441",
        "LIDC_IDRI_0490",
        "LIDC_IDRI_0529",
        "LIDC_IDRI_0606",
        "LIDC_IDRI_0621",
        "LIDC_IDRI_0648",
        "LIDC_IDRI_0651",
        "LIDC_IDRI_0657",
        "LIDC_IDRI_0663",
        "LIDC_IDRI_0673",
        "LIDC_IDRI_0676",
        "LIDC_IDRI_0684",
        "LIDC_IDRI_0696",
        "LIDC_IDRI_0698",
        "LIDC_IDRI_0710",
        "LIDC_IDRI_0722",
        "LIDC_IDRI_0744",
        "LIDC_IDRI_0757",
        "LIDC_IDRI_0778",
        "LIDC_IDRI_0784",
        "LIDC_IDRI_0810",
        "LIDC_IDRI_0813",
        "LIDC_IDRI_0819",
        "LIDC_IDRI_0831",
        "LIDC_IDRI_0837",
        "LIDC_IDRI_0856",
        "LIDC_IDRI_0874",
        "LIDC_IDRI_0876",
        "LIDC_IDRI_0909",
        "LIDC_IDRI_0920",
        "LIDC_IDRI_0981",
        "LIDC_IDRI_1001",
        "LIDC_IDRI_1004",
    ]
    if train_names is None and test_names is None:
        assert False
    if train_names is None:
        train_names = [name for name in names if name not in test_names]
    i = 0
    new_list = []
    new_dict = {}
    for key, v in exact_embeddings_dict.items():
        # print(key,key[:14])
        if key[:14] in train_names or key in train_names:
            new_list.append(exact_embeddings_list[i])
            new_dict[key] = exact_embeddings_dict[key]
        i += 1
    exact_embeddings_list = new_list
    exact_embeddings_dict = new_dict

    i = 0
    new_list = []
    new_dict = {}
    for key, v in lidc_embeddings_dict.items():
        if key[:14] in train_names or key in train_names:
            new_list.append(lidc_embeddings_list[i])
            new_dict[key] = lidc_embeddings_dict[key]
        i += 1
    lidc_embeddings_list = new_list
    lidc_embeddings_dict = new_dict

    exact_stacked_embeddings_numpy = np.stack(exact_embeddings_list, axis=0)
    lidc_stacked_embeddings_numpy = np.stack(lidc_embeddings_list, axis=0)

    exact_lidc_concatenated_array = np.concatenate(
        (exact_stacked_embeddings_numpy, lidc_stacked_embeddings_numpy), axis=0
    )
    merged_dict = {**exact_embeddings_dict, **lidc_embeddings_dict}
    merged_list = list(exact_embeddings_dict.keys()) + list(lidc_embeddings_dict.keys())

    return exact_lidc_concatenated_array, merged_dict, merged_list
