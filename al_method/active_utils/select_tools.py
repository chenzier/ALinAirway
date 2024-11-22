import torch
from torch import from_numpy as from_numpy


def select_from_candidates(
    select_num, candidates_array, candidates_dict, candidates_list, device
):
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
