import json
import random

def extract_random_samples_from_jsonl(input_file_path, output_file_path, num_samples=50000):
    """
    从JSONL文件中随机抽取指定数量的数据条目并保存到新文件中
    
    Args:
        input_file_path (str): 输入JSONL文件的路径
        output_file_path (str): 输出JSONL文件的路径
        num_samples (int): 要抽取的样本数量，默认为50000
    
    Returns:
        int: 实际抽取并写入的条目数量
    """
    lines = []
    total_lines = 0
    
    try:
        # 首先计算总行数
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                if line.strip():
                    total_lines += 1
        
        print(f"文件总共有 {total_lines} 条数据")
        
        # 如果请求数量大于等于总数，则复制整个文件
        if num_samples >= total_lines:
            print("请求数量大于等于文件总行数，将复制整个文件")
            with open(input_file_path, 'r', encoding='utf-8') as input_file, \
                 open(output_file_path, 'w', encoding='utf-8') as output_file:
                
                count = 0
                for line in input_file:
                    if line.strip():
                        output_file.write(line)
                        count += 1
                
                return count
        
        # 随机选择行号
        selected_line_numbers = set(random.sample(range(1, total_lines + 1), num_samples))
        
        # 读取选定的行
        with open(input_file_path, 'r', encoding='utf-8') as input_file, \
             open(output_file_path, 'w', encoding='utf-8') as output_file:
            
            line_number = 0
            written_count = 0
            
            for line in input_file:
                if line.strip():
                    line_number += 1
                    if line_number in selected_line_numbers:
                        output_file.write(line)
                        written_count += 1
                        
                        # 提前结束如果已经写入足够数量
                        if written_count >= num_samples:
                            break
            
            return written_count
                        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}")
        return 0
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return 0

# 使用示例
input_file_path = "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/retrieval_engine/retrieved_results/train.jsonl"
output_file_path = "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/retrieval_engine/retrieved_results/train_triviaqa_random_sample_5k.jsonl"

# 设置随机种子以便结果可重现（可选）
random.seed(42)

extracted_count = extract_random_samples_from_jsonl(input_file_path, output_file_path, 5000)
print(f"成功随机抽取并保存了 {extracted_count} 条数据到 {output_file_path}")