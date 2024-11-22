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
