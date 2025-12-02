import json
import math

# 输入文件路径，包含训练数据的JSON文件
input_file = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/nq_data_train.json'
# 输出文件路径，处理后的数据将保存为JSONL格式
output_file = './nq_train_selector_ppl_log.jsonl'

# 读取输入文件中的JSON数据
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 处理数据：对每个样本按PPL_CD排序，提取正负样本及对应分数
new_data = []
for item in data:
    question = item['question']
    answers = item['golden_answers']
    passages = item['passages']
    
    # 根据PPL_CD值对passages进行升序排序
    sorted_passages = sorted(passages, key=lambda x: x["PPL_CD"])
    
    # 构造文本列表，每个元素是title和text的拼接
    texts = [passage['title']+'\n'+passage['text'] for passage in sorted_passages]
    
    # 计算每个passage的得分（使用-PPL_CD的对数变换）
    scores= [-math.log(passage['PPL_CD']+1) for passage in sorted_passages]
    
    # 构建新的数据项
    new_item = dict()
    new_item['query'] = question
    new_item['pos'] = texts[0:1]          # 第一个为正样本
    new_item['neg'] = texts[1:]           # 其余为负样本
    new_item['pos_scores'] = scores[0:1]  # 正样本得分
    new_item['neg_scores'] = scores[1:]   # 负样本得分
    
    # 设置prompt字段，用于指导模型判断passage是否包含query的答案
    new_item['prompt'] = "Given a query A and a passage B, determine whether the passage directly or indirectly contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
    new_data.append(new_item)

# 将处理后的数据写入输出文件，每行一个JSON对象
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(json.dumps(record, ensure_ascii=False) + '\n' for record in new_data)