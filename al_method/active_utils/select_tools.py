import torch
from torch import from_numpy as from_numpy
import pickle
import random
from .file_tools import save_obj, load_obj


def extract_random_num_percent_by_key(num, dictionary):  # 随机抽取字典中num个样本
    keys = list(dictionary.keys())
    random.shuffle(keys)
    randon_dict = {key: dictionary[key] for key in keys}

    num_elements = len(keys)
    num_to_extract = int(num_elements * num)
    top_num_percent_keys = keys[:num_to_extract]
    return {key: randon_dict[key] for key in top_num_percent_keys}


def select_from_uncertainy(
    uncertainy_path, data_dict1, data_dict2, num, save_path=None
):
    # 对于negtive,抽取m%的样本
    def extract_random_num_percent_by_key(num, dictionary):  # 随机抽取字典中num个样本
        keys = list(dictionary.keys())
        random.shuffle(keys)
        randon_dict = {key: dictionary[key] for key in keys}

        num_elements = len(keys)
        num_to_extract = int(num_elements * num)
        top_num_percent_keys = keys[:num_to_extract]
        return {key: randon_dict[key] for key in top_num_percent_keys}

    result_dict = extract_random_num_percent_by_key(num, data_dict2)
    # 计算从uncertainy抽取的样本数num3
    num1 = int(num * len(data_dict1))
    num2 = len(result_dict)
    num3 = num1 - num2

    # 读取uncertainy_dict

    loaded_data = load_obj(uncertainy_path)
    uncertainy_dict = loaded_data["uncertainy_dict"]
    print(len(uncertainy_dict))
    # 将uncertainy升序排序
    sorted_dict = dict(sorted(uncertainy_dict.items(), key=lambda item: item[1]))
    sorted_list = list(sorted_dict.keys())

    # 抽取前sample_number个样本并放入result_dict
    al_list = sorted_list[:num3]
    for i in al_list:
        print(i)
        temp = i[:-7]
        result_dict[temp] = data_dict1[temp]

    # 使用save_obj函数保存
    if save_path is not None:
        save_obj(result_dict, save_path)
    return num1, num2, num3


def select_from_candidates(select_num, candidates_array, candidates_list, device):
    def sim(a, b):
        # 调整输入张量的维度
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)

        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return cos_sim

    def get_score2(i, score1s, score2s):

        # 将 float('-inf') 的数据类型设置为与 score1s 和 score2s 一致
        negative_inf = torch.tensor(
            float("-inf"), dtype=score1s.dtype, device=score1s.device
        )
        # 计算score2s，保持已经是负无穷大的值不变
        new_score2s = torch.where(
            score1s == negative_inf, negative_inf, (i + 1) / score1s
        )
        # 检查是否有 score2s 中的值已经是负无穷大，如果是，则保持不变
        score2s = torch.where(score2s == negative_inf, negative_inf, new_score2s)
        # 获取最大值对应的索引
        new_candidate = torch.argmax(score2s).item()
        return score2s, new_candidate

    candidates_tensor = from_numpy(candidates_array)
    candidates_tensor = candidates_tensor.to(device)

    scores_shape = candidates_tensor.shape[0]
    score1s = torch.ones(scores_shape, device=candidates_tensor.device)
    score2s = torch.zeros(scores_shape, device=candidates_tensor.device)
    selcet_list = []

    new_candidate = 0
    score2s[new_candidate] = float("-inf")
    new_candidate_representiveness = candidates_tensor[new_candidate]
    selcet_list.append(candidates_list[new_candidate])
    for i in range(select_num - 1):
        print(f"epoch {i}")
        # 计算相似度
        sim_output = sim(new_candidate_representiveness, candidates_tensor)
        # print(sim_output.shape)
        score1s = score1s + sim_output

        # 调用你的 get_score2 函数
        score2s, new_candidate = get_score2(i, score1s, score2s)
        # print('new_candidate',new_candidate,candidates_list[new_candidate])
        selcet_list.append(candidates_list[new_candidate])
        score2s[new_candidate] = float("-inf")
        new_candidate_representiveness = candidates_tensor[new_candidate]

    return selcet_list
