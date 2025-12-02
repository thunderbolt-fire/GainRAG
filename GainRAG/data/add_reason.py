import json
from collections import defaultdict

# 输入文件路径
reason_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/RADIO/dataset/nq/reason/train_with_reason.jsonl'
target_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/data_train_nq_filter.json'

# 输出文件路径
output_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/data_train_nq_filter_with_reason.json'

# 读取包含reason的数据
print("正在读取reason数据...")
question_to_reason = {}

with open(reason_file, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        try:
            item = json.loads(line.strip())
            question = item.get('question', '').strip()
            reason = item.get('reason', '')
            
            if question and reason:
                question_to_reason[question] = reason
        except json.JSONDecodeError as e:
            print(f"解析第{line_num}行时出错: {e}")
            continue

print(f"从reason文件中读取了 {len(question_to_reason)} 个问题的reason")

# 读取目标数据文件
print("正在读取目标数据文件...")
with open(target_file, 'r', encoding='utf-8') as f:
    target_data = json.load(f)

print(f"目标文件包含 {len(target_data)} 个样本")

# 添加reason到目标数据
matched_count = 0
unmatched_count = 0

for item in target_data:
    question = item.get('question', '').strip()
    
    if question in question_to_reason:
        item['reason'] = question_to_reason[question]
        matched_count += 1
    else:
        item['reason'] = ""  # 如果没有找到对应的reason，设置为空字符串
        unmatched_count += 1

print(f"匹配到reason的问题: {matched_count}")
print(f"未匹配到reason的问题: {unmatched_count}")
print(f"匹配率: {matched_count / len(target_data) * 100:.2f}%")

# 保存结果
print("正在保存结果...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(target_data, f, ensure_ascii=False, indent=2)

print(f"结果已保存到: {output_file}")

# 输出一些统计信息
print("\n统计信息:")
print(f"- 原始数据: {len(target_data)} 个样本")
print(f"- Reason数据: {len(question_to_reason)} 个问题")
print(f"- 成功匹配: {matched_count} 个")
print(f"- 未匹配: {unmatched_count} 个")

# 显示一些示例数据
print("\n示例数据(前3个):")
for i, item in enumerate(target_data[:3]):
    print(f"样本 {i+1}:")
    print(f"  问题: {item.get('question', '')[:100]}...")
    print(f"  是否有reason: {'是' if item.get('reason') else '否'}")
    if item.get('reason'):
        print(f"  reason: {item.get('reason')[:100]}...")
    print()