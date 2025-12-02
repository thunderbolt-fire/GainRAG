import json
import os




merged_data = list()


filepath = '/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/data_train_nq_filter_with_reason_scores.json'

# Get the base filename and add _filter to it
base_filename = os.path.basename(filepath)
filename_without_ext, file_extension = os.path.splitext(base_filename)
output_file = f'./without_pse_{filename_without_ext}_filter{file_extension}'



with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = list()
for item in data:
    passages = item['passages']
    # 去掉最后一个passage
    if len(passages) > 0:
        item['passages'] = passages[:-1]
    
    sorted_passages = sorted(item['passages'], key=lambda x: x["PPL_CD"])
    silver_passage = sorted_passages[0]
    if silver_passage['EM_with_retrieval']:
        new_data.append(item)
print( ': ',len(new_data))
merged_data.extend(new_data)
print('data items:', len(merged_data))

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)
