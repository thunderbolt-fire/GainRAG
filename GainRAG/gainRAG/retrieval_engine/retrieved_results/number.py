import json

def analyze_jsonl_file(file_path):
    """
    分析JSONL文件，计算条目数量并提供详细信息
    
    Args:
        file_path (str): JSONL文件的路径
    
    Returns:
        dict: 包含分析结果的字典
    """
    count = 0
    valid_entries = 0
    invalid_entries = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line.strip():  # 确保不是空行
                    count += 1
                    try:
                        # 尝试解析JSON以验证其有效性
                        json.loads(line)
                        valid_entries += 1
                    except json.JSONDecodeError:
                        invalid_entries += 1
                        print(f"警告：第{line_num}行不是有效的JSON格式")
                        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None
    
    return {
        "total_entries": count,
        "valid_entries": valid_entries,
        "invalid_entries": invalid_entries
    }

# 使用示例
file_path = "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/retrieval_engine/retrieved_results/train.jsonl"
result = analyze_jsonl_file(file_path)

if result:
    print(f"文件分析完成:")
    print(f"  总条目数: {result['total_entries']}")
    print(f"  有效条目: {result['valid_entries']}")
    print(f"  无效条目: {result['invalid_entries']}")