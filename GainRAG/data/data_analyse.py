# %%

import matplotlib.pyplot as plt
import numpy as np
import json

# 1. 配置：指定要可视化的数据行号 (从0开始)
# 例如: [0] 只看第一条，[0, 5, 10] 看第1、6、11条， range(0, 3) 看前3条
indices_to_visualize = [0, 1] 
file_path = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_nq_train_selector_debug_data.jsonl'

def plot_record(record, index):
    """封装绘图逻辑"""
    # 提取 debug_info
    if "debug_info" in record:
        data = record["debug_info"]
    else:
        print(f"Warning: 'debug_info' not found in record {index}.")
        data = record

    if not data:
        return

    # 提取数据
    combined = data.get('combined_scores', [])
    passages = np.arange(len(combined))
    
    # 归一化分数
    retrieval = data.get('retrieval_normalized', [])
    ppl = data.get('ppl_normalized', [])
    reason = data.get('reason_normalized', [])
    
    # 原始分数 & 中间分数
    retrieval_orig = data.get('retrieval_original', [])
    ppl_orig = data.get('ppl_original', [])
    ppl_softmax = data.get('ppl_softmax', [])
    reason_orig = data.get('reason_original', [])

    # 判断是否有 reason 数据
    has_reason = len(reason) == len(passages) and len(reason) > 0

    # --- 设置绘图布局 ---
    bottom_cols = 4 if has_reason else 3
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, bottom_cols)

    # ==========================================
    # 第一行: 归一化分数与组合分数
    # ==========================================
    ax_norm = fig.add_subplot(gs[0, :])
    
    if has_reason:
        bar_width = 0.25
        offset_retrieval = -bar_width
        offset_ppl = 0
        offset_reason = bar_width
    else:
        bar_width = 0.35
        offset_retrieval = -bar_width/2
        offset_ppl = bar_width/2
        offset_reason = 0

    if len(retrieval) == len(passages):
        ax_norm.bar(passages + offset_retrieval, retrieval, width=bar_width, label='Retrieval (Norm)', color='#88c999', alpha=0.8)
    if len(ppl) == len(passages):
        ax_norm.bar(passages + offset_ppl, ppl, width=bar_width, label='PPL (Norm)', color='#8899c9', alpha=0.8)
    if has_reason:
        ax_norm.bar(passages + offset_reason, reason, width=bar_width, label='Reason (Norm)', color='#c98888', alpha=0.8)
    if len(combined) > 0:
        ax_norm.plot(passages, combined, label='Combined Score', color='red', linewidth=2, marker='o', linestyle='-')

    query_text = record.get('query', 'Unknown Query')
    ax_norm.set_title(f'Record #{index}: "{query_text}"', fontsize=14)
    ax_norm.set_ylabel('Normalized Score (0-1)')
    ax_norm.set_xticks(passages)
    ax_norm.legend(loc='upper right')
    ax_norm.grid(axis='y', linestyle='--', alpha=0.5)

    # ==========================================
    # 第二行: 原始分数与中间分数
    # ==========================================
    # 2.1 Retrieval Original
    ax_ret_orig = fig.add_subplot(gs[1, 0])
    if len(retrieval_orig) == len(passages):
        ax_ret_orig.bar(passages, retrieval_orig, color='#88c999', alpha=0.6)
        ax_ret_orig.set_title('Original Retrieval Score')
        ax_ret_orig.set_xticks(passages)
        ax_ret_orig.grid(axis='y', linestyle='--', alpha=0.3)

    # 2.2 PPL Original (Log Scale)
    ax_ppl_orig = fig.add_subplot(gs[1, 1])
    if len(ppl_orig) == len(passages):
        ax_ppl_orig.bar(passages, ppl_orig, color='#8899c9', alpha=0.6)
        ax_ppl_orig.set_title('Original PPL (Log Scale)')
        ax_ppl_orig.set_yscale('log') 
        ax_ppl_orig.set_xticks(passages)
        ax_ppl_orig.grid(axis='y', linestyle='--', alpha=0.3)

    # 2.3 PPL Softmax
    ax_ppl_soft = fig.add_subplot(gs[1, 2])
    if len(ppl_softmax) == len(passages):
        ax_ppl_soft.bar(passages, ppl_softmax, color='#9988c9', alpha=0.6)
        ax_ppl_soft.set_title('PPL Softmax (Probability)')
        ax_ppl_soft.set_xticks(passages)
        ax_ppl_soft.grid(axis='y', linestyle='--', alpha=0.3)

    # 2.4 Reason Original (如果有)
    if has_reason:
        ax_reason_orig = fig.add_subplot(gs[1, 3])
        if len(reason_orig) == len(passages):
            ax_reason_orig.bar(passages, reason_orig, color='#c98888', alpha=0.6)
            ax_reason_orig.set_title('Original Reason Score')
            ax_reason_orig.set_xticks(passages)
            ax_reason_orig.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

# 2. 读取文件并处理指定行
try:
    target_indices_set = set(indices_to_visualize)
    max_index = max(target_indices_set) if target_indices_set else -1
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in target_indices_set:
                try:
                    record = json.loads(line)
                    print(f"Visualizing Record Index: {i}")
                    plot_record(record, i)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON at line {i}")
            
            # 如果已经处理完所有需要的行，提前退出循环
            if i >= max_index:
                break
                
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")