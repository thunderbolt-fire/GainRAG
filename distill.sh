#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 使用 nohup 运行训练命令
nohup torchrun --nproc_per_node=1 --master_port=29500 \
-m FlagEmbedding.finetune.reranker.encoder_only.base \
--output_dir ./model_outputs/r0.7_p0.3 \
--model_name_or_path BAAI/bge-reranker-base \
--train_data /root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/20250922_091953_without_pse_data_train_nq_filter_softmax_combined_r0.7_p0.3.jsonl \
--overwrite_output_dir \
--train_group_size 16 \
--knowledge_distillation True \
--query_max_len 256 \
--passage_max_len 256 \
--max_len 512 \
--pad_to_multiple_of 8 \
--learning_rate 6e-5 \
--fp16 \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--logging_steps 10 \
--save_steps 5000 \
--save_strategy steps \
--dataloader_num_workers 0 \
--remove_unused_columns False \
> distill_training.log 2>&1 &

echo "Training started in background. Check progress with: tail -f distill_training.log"
echo "Process ID: $!"

# explanations of arguments
# train_group_size: number of positive and negative samples
# resume_from_checkpoint: path to the checkpoint to resume from
# save_strategy: no, steps, or epoch
# torchrun with nproc_per_node=1 initializes distributed environment for single GPU
# reduced query_max_len and passage_max_len to avoid sequence length issues
# max_len set to 512 which is within model's positional encoding limit
# reduced batch_size further and increased gradient_accumulation_steps to avoid memory issues
# added dataloader_num_workers=0 to avoid multiprocessing issues
# nohup allows the process to continue running after terminal disconnection
# > distill_training.log 2>&1 & redirects all output to log file and runs in background