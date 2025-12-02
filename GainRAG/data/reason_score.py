import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import torch

def load_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def compute_reason_similarity_scores(data, device='cuda:0'):
    """
    为每个passage计算与reason的相似度分数
    只处理有reason的数据，没有reason的数据会被过滤掉
    
    Args:
        data: 包含question, reason, passages的数据列表
        device: 使用的设备
    
    Returns:
        处理后的数据，每个passage增加了reason_similarity_score字段
    """
    # 加载嵌入模型
    print("正在加载BGE-M3嵌入模型...")
    embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device)
    
    processed_data = []
    
    for item in tqdm(data, desc="计算reason相似度分数"):
        question = item.get('question', '')
        reason = item.get('reason', '')
        passages = item.get('passages', [])
        golden_answers = item.get('golden_answers', [])
        
        # 只处理有reason的数据
        if not reason or not reason.strip():
            continue  # 跳过没有reason的样本
        
        # 构造reason文本：question + golden_answers + reason
        reason_text = f"{question}. {', '.join(golden_answers)}. {reason}"
        
        # 提取passage文本
        passage_texts = []
        for passage in passages:
            title = passage.get('title', '')
            text = passage.get('text', '')
            passage_content = f"{title}\n{text}" if title else text
            passage_texts.append(passage_content)
        
        if passage_texts:
            # 计算reason和passages的嵌入
            reason_embedding = embedding_model.encode([reason_text])['dense_vecs']
            passages_embeddings = embedding_model.encode(passage_texts)['dense_vecs']
            
            # 计算相似度分数
            similarity_scores = (reason_embedding @ passages_embeddings.T).flatten()
            
            # 为每个passage添加相似度分数
            for i, passage in enumerate(passages):
                passage['reason_similarity_score'] = float(similarity_scores[i])
        
        # 添加到处理后的数据中
        processed_item = item.copy()
        processed_data.append(processed_item)
    
    return processed_data

def main():
    # 输入文件路径
    input_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/data_train_nq_filter_with_reason.json'
    
    # 输出文件路径
    output_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/data_train_nq_filter_with_reason_scores.json'
    
    # 设备配置
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("正在加载数据...")
    data = load_data(input_file)
    print(f"原始数据包含 {len(data)} 个样本")
    
    # 计算有reason的样本数量
    has_reason_count = sum(1 for item in data if item.get('reason', '').strip())
    no_reason_count = len(data) - has_reason_count
    
    print(f"有reason的样本数量: {has_reason_count}")
    print(f"没有reason的样本数量: {no_reason_count}")
    print(f"有reason的比例: {has_reason_count/len(data)*100:.2f}%")
    
    # 计算相似度分数（只处理有reason的数据）
    print("开始计算reason相似度分数（只处理有reason的数据）...")
    processed_data = compute_reason_similarity_scores(data, device=device)
    
    print(f"处理后保留的样本数量: {len(processed_data)}")
    print(f"剔除的样本数量: {len(data) - len(processed_data)}")
    
    # 计算分数统计信息
    all_scores = []
    total_passages = 0
    for item in processed_data:
        for passage in item.get('passages', []):
            if 'reason_similarity_score' in passage:
                all_scores.append(passage['reason_similarity_score'])
                total_passages += 1
    
    print(f"处理的passages总数: {total_passages}")
    
    if all_scores:
        print(f"\nReason相似度分数统计:")
        print(f"- 分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}]")
        print(f"- 平均分数: {np.mean(all_scores):.6f}")
        print(f"- 标准差: {np.std(all_scores):.6f}")
    
    # 保存结果
    print("正在保存结果...")
    save_data(processed_data, output_file)
    print(f"结果已保存到: {output_file}")
    
    # 显示示例
    print("\n示例数据:")
    for i, item in enumerate(processed_data[:2]):
        print(f"\n样本 {i+1}:")
        print(f"  问题: {item.get('question', '')[:100]}...")
        print(f"  Reason: {item.get('reason', '')[:100]}...")
        
        passages = item.get('passages', [])
        if passages:
            print(f"  Passages数量: {len(passages)}")
            for j, passage in enumerate(passages[:3]):  # 只显示前3个
                score = passage.get('reason_similarity_score', 'N/A')
                title = passage.get('title', '')[:50]
                print(f"    Passage {j+1}: {title}... (reason相似度: {score})")

if __name__ == "__main__":
    main()