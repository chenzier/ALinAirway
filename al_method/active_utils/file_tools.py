import pickle
import time
import subprocess
import os

"""
    加载/读取 pkl文件
"""


def save_obj(obj, name):
    if name[-3:] != "pkl":
        temp = name + ".pkl"
    else:
        temp = name
    with open(temp, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if name[-3:] != "pkl":
        temp = name + ".pkl"
    else:
        temp = name
    # print(temp)
    with open(temp, "rb") as f:
        return pickle.load(f)


def save_in_chunks(embeddings_dict, file_path_base, chunk_size=2000):
    # 确保文件夹存在，如果不存在则创建它
    os.makedirs(file_path_base, exist_ok=True)

    # 获取embeddings_dict的键值对列表，方便后续分块处理
    embeddings_dict_items = list(embeddings_dict.items())

    # 分块保存逻辑
    num_chunks = len(embeddings_dict_items) // chunk_size + (
        1 if len(embeddings_dict_items) % chunk_size != 0 else 0
    )
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings_dict_items))
        chunk_data = dict(embeddings_dict_items[start_idx:end_idx])

        file_path = os.path.join(file_path_base, f"chunk_{i}.pkl")
        save_obj({"embedding_chunk": chunk_data}, file_path)
        print(f"{i / num_chunks} is done")
    print("all done")


def load_chunks(file_path):
    """
    加载指定路径下以chunk_开头的.pkl文件，并合并其中的数据。

    Args:
        file_path (str): 要加载文件所在的路径，例如 "/data/wangc/al_data/test1123/embedding/exact09_128_op"

    Returns:
        dict: 合并后的字典数据，包含了从各个分块文件中加载的数据。
    """
    res_dict = {}
    for file_name in os.listdir(file_path):
        if file_name.startswith("chunk_") and file_name.endswith(".pkl"):
            file_full_path = os.path.join(file_path, file_name)
            with open(file_full_path, "rb") as file:
                data = pickle.load(file)
                res_dict.update(data["embedding_chunk"])
    return res_dict


"""
    将原数据集的一部分文件(子集)取出，放入目的文件夹中
"""


# generate_folder_for_selected(source_folder, target_folder, names_filtered, num=50)
def generate_folder_for_selected(
    source,  # 原数据集位置
    target,  # 目的路径
    selected_name,  # 要进行选择的name
    num=None,
):  # 如果只是想看一部分，而不是选择全部，设置num即可

    address1 = ["/image", "/label"]

    for adr in address1:
        source_path = source + adr
        target_path = target + adr
        os.makedirs(target_path, exist_ok=True)
        if num is None:  # 如果 num 未指定，默认为选定名称的长度
            num = len(selected_name)
        for i in range(num):
            print(f"{i/num*100:.2f}%", end="\r")
            time.sleep(0.5)  # 模拟耗时操作
            file_name = selected_name[i]
            source_file = os.path.join(source_path, file_name)
            target_file = os.path.join(target_path, file_name)
            subprocess.run(["cp", source_file, target_file])  ## 复制文件到目标路径
