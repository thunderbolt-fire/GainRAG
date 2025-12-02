import json
import math
import numpy as np
import os

# 输入文件路径，包含reason分数的训练数据
input_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_data_train_nq_filter_with_reason_scores_filter.json'

# 动态生成输出文件名：在输入文件名基础上添加后缀
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f'./{base_name}_softmax_combined.jsonl'
debug_file = f'./{base_name}_debug_data.jsonl'

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

def compute_combined_scores_with_reason(passages, retrieval_weight=0.33, ppl_weight=0.33, reason_weight=0.34):
    """
    结合retrieval_score、PPL_CD和reason_similarity_score，使用最大最小值归一化后加权求和
    
    Args:
        passages: 包含retrieval_score、PPL_CD和reason_similarity_score的passage列表
        retrieval_weight: retrieval_score的权重
        ppl_weight: PPL_CD的权重
        reason_weight: reason_similarity_score的权重
    
    Returns:
        combined_scores: 加权求和后的分数列表
        debug_info: 包含所有中间处理步骤的调试信息
    """
    # 检查是否有reason分数
    has_reason_scores = any('reason_similarity_score' in passage for passage in passages)
    
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
        ppl = passage.get('PPL_CD', 0.0)
        try:
            ppl_value = float(ppl)
            ppl_original.append(ppl_value)
            # 对PPL进行负对数变换（PPL越小越好，所以取负对数）
            transformed_score = -math.log(ppl_value + 1)
            ppl_scores.append(transformed_score)
        except (ValueError, TypeError):
            ppl_original.append(0.0)
            ppl_scores.append(0.0)  # 使用默认值
    
    # 提取reason相似度分数（如果存在）
    reason_scores = []
    if has_reason_scores:
        for passage in passages:
            score = passage.get('reason_similarity_score', 0.0)
            try:
                reason_scores.append(float(score))
            except (ValueError, TypeError):
                reason_scores.append(0.0)
    
    # 对变换后的PPL分数进行softmax归一化
    ppl_softmax_scores = softmax(ppl_scores)
    
    # 最大最小值归一化
    def min_max_normalize(scores):
        scores = np.array(scores, dtype=float)
        min_score = scores.min()
        max_score = scores.max()
        if max_score == min_score:
            return np.zeros_like(scores)  # 如果最大最小值相等，返回全零
        return (scores - min_score) / (max_score - min_score)
    
    # 对各类分数分别进行最大最小值归一化
    norm_retrieval = min_max_normalize(retrieval_scores)
    norm_ppl = min_max_normalize(ppl_softmax_scores)
    
    if has_reason_scores:
        norm_reason = min_max_normalize(reason_scores)
        # 三种分数加权组合
        combined_scores = (retrieval_weight * norm_retrieval + 
                         ppl_weight * norm_ppl + 
                         reason_weight * norm_reason)
    else:
        # 只有两种分数时，重新分配权重
        adjusted_retrieval_weight = retrieval_weight / (retrieval_weight + ppl_weight)
        adjusted_ppl_weight = ppl_weight / (retrieval_weight + ppl_weight)
        combined_scores = (adjusted_retrieval_weight * norm_retrieval + 
                         adjusted_ppl_weight * norm_ppl)
        norm_reason = []
    
    # 构建调试信息
    debug_info = {
        'has_reason_scores': has_reason_scores,
        'ppl_original': ppl_original,
        'ppl_transformed': ppl_scores,
        'ppl_softmax': ppl_softmax_scores.tolist(),
        'ppl_normalized': norm_ppl.tolist(),
        'retrieval_original': retrieval_scores,
        'retrieval_normalized': norm_retrieval.tolist(),
        'reason_original': reason_scores if has_reason_scores else [],
        'reason_normalized': norm_reason.tolist() if has_reason_scores else [],
        'combined_scores': combined_scores.tolist(),
        'weights_used': {
            'retrieval': retrieval_weight if has_reason_scores else adjusted_retrieval_weight,
            'ppl': ppl_weight if has_reason_scores else adjusted_ppl_weight,
            'reason': reason_weight if has_reason_scores else 0.0
        }
    }
    
    return combined_scores.tolist(), debug_info

# 处理数据：使用改进的综合分数计算方法
new_data = []
debug_data = []

# 统计有reason分数的样本数量
samples_with_reason = 0
samples_without_reason = 0

for item in data:
    question = item['question']
    answers = item['golden_answers']
    passages = item['passages']
    
    # 检查是否有reason分数
    has_reason = any('reason_similarity_score' in passage for passage in passages)
    if has_reason:
        samples_with_reason += 1
    else:
        samples_without_reason += 1
    
    # 计算综合分数并获取调试信息
    combined_scores, debug_info = compute_combined_scores_with_reason(
        passages, 
        retrieval_weight=0.33, 
        ppl_weight=0.33,
        reason_weight=0.34
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
        'has_reason_scores': debug_info['has_reason_scores'],
        'debug_info': debug_info
    }
    debug_data.append(debug_item)

# 输出统计信息
print(f"\n样本统计:")
print(f"- 有reason分数的样本: {samples_with_reason}")
print(f"- 没有reason分数的样本: {samples_without_reason}")
print(f"- 总样本数: {len(data)}")
print(f"- 有reason分数的比例: {samples_with_reason/len(data)*100:.2f}%")

# 验证分数范围
all_scores = []
for item in new_data:
    all_scores.extend(item['pos_scores'] + item['neg_scores'])

print(f"\n综合分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}]")
print(f"处理了 {len(new_data)} 个样本")

# 将处理后的数据写入输出文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in new_data)

print(f"数据已保存到: {output_file}")

# 将调试数据写入调试文件
with open(debug_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in debug_data)

print(f"调试数据已保存到: {debug_file}")

# 显示权重使用情况的统计
with_reason_count = sum(1 for item in debug_data if item['has_reason_scores'])
without_reason_count = len(debug_data) - with_reason_count

print(f"\n权重使用统计:")
print(f"- 使用三种分数融合的样本: {with_reason_count}")
print(f"- 使用两种分数融合的样本: {without_reason_count}")