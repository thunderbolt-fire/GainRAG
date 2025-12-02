#随机从/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_nq_train_selector_softmax_combined.jsonl中抽取13971个数据

import json
import random

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """保存数据为JSONL格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def random_sample_data(input_file, output_file, sample_size=13971, seed=42):
    """
    从输入文件中随机抽取指定数量的数据
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        sample_size: 抽取的样本数量
        seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(seed)
    
    # 加载数据
    print("正在加载数据...")
    data = load_jsonl(input_file)
    total_size = len(data)
    
    print(f"原始数据总数: {total_size}")
    print(f"需要抽取的数据量: {sample_size}")
    
    # 检查数据量是否足够
    if total_size < sample_size:
        print(f"警告: 原始数据量({total_size})小于需要抽取的数量({sample_size})")
        print("将使用全部数据")
        sampled_data = data
        actual_sample_size = total_size
    else:
        # 随机抽取数据
        print("正在进行随机抽样...")
        sampled_data = random.sample(data, sample_size)
        actual_sample_size = sample_size
    
    # 保存抽取的数据
    print("正在保存抽取的数据...")
    save_jsonl(sampled_data, output_file)
    
    print(f"成功抽取了 {actual_sample_size} 个样本")
    print(f"抽样比例: {actual_sample_size/total_size*100:.2f}%")
    print(f"数据已保存到: {output_file}")
    
    return sampled_data

def main():
    # 输入文件路径
    input_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_nq_train_selector_softmax_combined.jsonl'
    
    # 输出文件路径
    output_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_nq_train_selector_sampled_13971.jsonl'
    
    # 抽取数据
    sampled_data = random_sample_data(
        input_file=input_file,
        output_file=output_file,
        sample_size=13971,
        seed=42  # 固定随机种子确保结果可复现
    )
    
    # 显示一些统计信息
    print("\n样本统计信息:")
    
    # 统计正负样本数量
    total_pos = sum(len(item.get('pos', [])) for item in sampled_data)
    total_neg = sum(len(item.get('neg', [])) for item in sampled_data)
    
    print(f"- 总查询数量: {len(sampled_data)}")
    print(f"- 正样本总数: {total_pos}")
    print(f"- 负样本总数: {total_neg}")
    print(f"- 平均每个查询的负样本数量: {total_neg/len(sampled_data):.2f}")
    
    # 显示分数范围
    all_pos_scores = []
    all_neg_scores = []
    
    for item in sampled_data:
        all_pos_scores.extend(item.get('pos_scores', []))
        all_neg_scores.extend(item.get('neg_scores', []))
    
    if all_pos_scores:
        print(f"- 正样本分数范围: [{min(all_pos_scores):.6f}, {max(all_pos_scores):.6f}]")
    if all_neg_scores:
        print(f"- 负样本分数范围: [{min(all_neg_scores):.6f}, {max(all_neg_scores):.6f}]")
    
    # 显示前3个样本示例
    print("\n前3个样本示例:")
    for i, item in enumerate(sampled_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  查询: {item.get('query', '')[:100]}...")
        print(f"  正样本数量: {len(item.get('pos', []))}")
        print(f"  负样本数量: {len(item.get('neg', []))}")
        if item.get('pos_scores'):
            print(f"  正样本分数: {item['pos_scores']}")
        if item.get('neg_scores'):
            print(f"  负样本分数前3个: {item['neg_scores'][:3]}")

if __name__ == "__main__":
    main()