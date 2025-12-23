#!/bin/bash

# 定义模型路径数组
model_paths=(
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_all_kd1_grpo0.1_kl0.1"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_grpo_1.0_kd_kl"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_grpo0.1_kl"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_kd1_grpo0.1_kl0.1"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_oll_KD"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_olly_grpo_0.1"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_grpo_new_sample_olly_grpo_1.0"
  "/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/gainRAG/model_outputs/test_old_sample_grpo0.1_kd1_kl0.1"
)

# 遍历每个模型路径
for model_path in "${model_paths[@]}"; do
  echo "=================================================="
  echo "Processing model: $model_path"
  echo "=================================================="
  
  # 提取目录名作为日志文件名
  dir_name=$(basename "$model_path")
  log_file="results_${dir_name}.log"
  
  # 运行Python指令并将结果保存到日志文件
  echo "Running command: python /root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/RADIO/run_rag.py --rerank_model_path $model_path"
  python /root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/RADIO/run_rag.py --rerank_model_path "$model_path" 2>&1 | tee "$log_file"
  
  echo "Results saved to: $log_file"
  echo ""
done

echo "All models processed."