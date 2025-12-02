import json
import math
import numpy as np
import os
from scipy.stats import rankdata
from datetime import datetime

# 输入文件路径，包含reason分数的训练数据
input_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_data_train_nq_filter.json'

# 动态生成输出文件名：在输入文件名基础上添加日期时间前缀
base_name = os.path.splitext(os.path.basename(input_file))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'./{timestamp}_{base_name}_softmax_combined.jsonl'
debug_file = f'./{timestamp}_{base_name}_debug_data.jsonl'

print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")
print(f"调试文件: {debug_file}")

# 读取输入文件中的JSON数据
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

def softmax(x):
    """计算softmax，确保数值稳定性"""
    x = np.array(x)
    # 减去最大值防止溢出
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def compute_combined_scores_with_penalty(passages, retrieval_weight=0.5, ppl_weight=0.5, lambda_param=0.05, gamma=5.0, use_penalty=True):
    """
    结合retrieval_score和PPL_CD，使用最大最小值归一化后加权求和，并可选择性添加基于排名的相对位置惩罚项
    
    Args:
        passages: 包含retrieval_score和PPL_CD的passage列表
        retrieval_weight: retrieval_score的权重
        ppl_weight: PPL_CD的权重
        lambda_param: 惩罚项强度参数
        gamma: 惩罚项敏感度参数
        use_penalty: 是否使用惩罚项，默认为True
    
    Returns:
        combined_scores: 加权求和后的分数列表
        debug_info: 包含所有中间处理步骤的调试信息
    """
    n = len(passages)
    
    # 提取retrieval分数（越大越好）
    retrieval_scores = []
    for passage in passages:
        score = passage.get('retrieval_score', 0.0)
        try:
            retrieval_scores.append(float(score))
        except (ValueError, TypeError):
            retrieval_scores.append(0.0)
    
    # 提取PPL分数并进行负对数变换
    ppl_original = []
    ppl_scores = []
    for passage in passages:
        ppl = passage['PPL_CD']
        try:
            ppl_value = float(ppl)
            ppl_original.append(ppl_value)
            # 对PPL进行负对数变换（PPL越小越好，所以取负对数）
            transformed_score = -math.log(ppl_value + 1)
            ppl_scores.append(transformed_score)
        except (ValueError, TypeError):
            ppl_original.append(0.0)
            ppl_scores.append(0.0)  # 使用默认值
    
    # 对变换后的PPL分数进行softmax归一化
    ppl_softmax_scores = softmax(ppl_scores)
    
    # 最大最小值归一化（参照score_cache_to_dataset.py）
    def min_max_normalize(scores):
        scores = np.array(scores, dtype=float)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)  # 如果最大最小值相等，返回全零
        return (scores - min_score) / (max_score - min_score)
    
    # 对retrieval分数和softmax后的PPL分数分别进行最大最小值归一化
    norm_retrieval = min_max_normalize(retrieval_scores)
    norm_ppl = min_max_normalize(ppl_softmax_scores)
    
    # 基础的加权组合分数
    combined_scores = retrieval_weight * norm_retrieval + ppl_weight * norm_ppl
    
    # 初始化惩罚相关变量
    rank_retrieval = None
    rank_ppl = None
    rank_diff = None
    penalty = None
    
    # 可选的惩罚项计算
    if use_penalty:
        # 计算排名（1表示最高分）
        # rankdata默认升序排列，分数越高排名越大，需要反转
        rank_retrieval = n + 1 - rankdata(retrieval_scores, method='ordinal')
        rank_ppl = n + 1 - rankdata(ppl_softmax_scores, method='ordinal')
        
        # 计算基于排名的相对位置惩罚项
        rank_diff = (rank_retrieval - rank_ppl) / n  # 归一化到[-1, 1]
        penalty = -lambda_param * np.tanh(gamma * rank_diff)
        
        # 应用惩罚项
        combined_scores = combined_scores - penalty
    else:
        # 不使用惩罚项时，创建零数组以保持调试信息完整性
        rank_retrieval = np.zeros(n)
        rank_ppl = np.zeros(n) 
        rank_diff = np.zeros(n)
        penalty = np.zeros(n)
    
    # 构建调试信息
    debug_info = {
        'ppl_original': ppl_original,
        'ppl_transformed': ppl_scores,
        'ppl_softmax': ppl_softmax_scores.tolist(),
        'ppl_normalized': norm_ppl.tolist(),
        'retrieval_original': retrieval_scores,
        'retrieval_normalized': norm_retrieval.tolist(),
        'rank_retrieval': rank_retrieval.tolist() if rank_retrieval is not None else [],
        'rank_ppl': rank_ppl.tolist() if rank_ppl is not None else [],
        'rank_diff': rank_diff.tolist() if rank_diff is not None else [],
        'penalty': penalty.tolist() if penalty is not None else [],
        'combined_scores': combined_scores.tolist(),
        'penalty_params': {'lambda': lambda_param, 'gamma': gamma, 'use_penalty': use_penalty}
    }
    
    return combined_scores.tolist(), debug_info

def compute_adaptive_penalty_params(retrieval_scores, ppl_normalized, rank_retrieval, rank_ppl):
    """
    自适应计算惩罚参数（基于排名）
    
    Args:
        retrieval_scores: 归一化后的检索分数
        ppl_normalized: 归一化后的PPL分数
        rank_retrieval: 检索分数排名
        rank_ppl: PPL分数排名
    
    Returns:
        lambda_param: 自适应的惩罚强度
        gamma: 自适应的敏感度参数
    """
    # 转换为NumPy数组
    rank_retrieval = np.array(rank_retrieval)
    rank_ppl = np.array(rank_ppl)
    n = len(rank_retrieval)
    
    # 基于排名差异的标准差来调整lambda
    rank_diff = (rank_retrieval - rank_ppl) / n
    std_rank_diff = np.std(rank_diff)
    
    # 根据排名差异的分布调整参数
    lambda_param = 0.4 * np.clip(std_rank_diff * 2, 0.5, 2.0)
    
    # 根据平均排名差异调整gamma
    mean_abs_rank_diff = np.mean(np.abs(rank_diff))
    if mean_abs_rank_diff > 0:
        gamma = 5.0 / (mean_abs_rank_diff + 0.1)
    else:
        gamma = 5.0
    
    # 限制参数范围
    lambda_param = np.clip(lambda_param, 0.2, 0.8)
    gamma = np.clip(gamma, 2.0, 10.0)
    
    return lambda_param, gamma

# 处理数据：使用改进的综合分数计算方法
new_data = []
debug_data = []

for item in data:
    question = item['question']
    answers = item['golden_answers']
    passages = item['passages']
    
    # 可以选择是否使用惩罚项
    # use_penalty=True: 使用惩罚项（默认）
    # use_penalty=False: 不使用惩罚项，只进行简单的加权求和
    combined_scores, debug_info = compute_combined_scores_with_penalty(
        passages, 
        retrieval_weight=0.7, 
        ppl_weight=0.3,
        lambda_param=0.01,
        gamma=5.0,
        use_penalty=False  # 设置为False可以禁用惩罚项
    )
    
    # 将分数添加到passages中并按分数降序排序
    for i, passage in enumerate(passages):
        passage['combined_score'] = combined_scores[i]
    
    sorted_passages = sorted(passages, key=lambda x: x["combined_score"], reverse=True)
    
    # 构造文本列表
    texts = [passage['title']+'\n'+passage['text'] for passage in sorted_passages]
    
    # 提取排序后的综合分数
    final_scores = [passage['combined_score'] for passage in sorted_passages]
    
    # 构建新的数据项
    new_item = dict()
    new_item['query'] = question
    new_item['pos'] = texts[0:1]          # 分数最高的为正样本
    new_item['neg'] = texts[1:]           # 其余为负样本
    new_item['pos_scores'] = final_scores[0:1]  # 正样本得分
    new_item['neg_scores'] = final_scores[1:]   # 负样本得分
    
    # 设置prompt字段
    new_item['prompt'] = "Given a query A and a passage B, determine whether the passage directly or indirectly contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
    new_data.append(new_item)
    
    # 保存调试数据
    debug_item = {
        'query': question,
        'num_passages': len(passages),
        'debug_info': debug_info
    }
    debug_data.append(debug_item)

# 验证分数范围
all_scores = []
for item in new_data:
    all_scores.extend(item['pos_scores'] + item['neg_scores'])

print(f"分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}]")
print(f"处理了 {len(new_data)} 个样本")

# 将处理后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in new_data)

print(f"数据已保存到: {output_file}")

# 将调试数据写入调试文件
with open(debug_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in debug_data)

print(f"调试数据已保存到: {debug_file}")