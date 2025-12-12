#!/bin/bash

# 定义变量
RERANKER_MODEL_PATH="/root/.cache/huggingface/hub/models--BAAI--bge-reranker-base/snapshots/2cfc18c9415c912f9d8155881c133215df768a70"
TRAIN_DATA_PATH="/root/siton-data-0553377b2d664236bad5b5d0ba8aa419/workspace/GainRAG/GainRAG/data/without_pse_nq_train_selector_sampled_13971.jsonl"
OUTPUT_DIR="./model_outputs/test_grpo2/"
TRAIN_GROUP_SIZE=16
QUERY_MAX_LEN=256
PASSAGE_MAX_LEN=256
MAX_LEN=512
PAD_TO_MULTIPLE_OF=8
LEARNING_RATE=6e-5
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
LOGGING_STEPS=1
SAVE_STEPS=5000
LOG_FILE="training.log"

# 训练命令并将日志同时输出到终端和文件
torchrun --nproc_per_node 1 -m selector_finetune \
    --model_name_or_path "${RERANKER_MODEL_PATH}" \
    --train_data "${TRAIN_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir \
    --train_group_size ${TRAIN_GROUP_SIZE} \
    --knowledge_distillation True \
    --query_max_len ${QUERY_MAX_LEN} \
    --passage_max_len ${PASSAGE_MAX_LEN} \
    --max_len ${MAX_LEN} \
    --pad_to_multiple_of ${PAD_TO_MULTIPLE_OF} \
    --learning_rate ${LEARNING_RATE} \
    --fp16 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --dataloader_drop_last True \
    --warmup_ratio ${WARMUP_RATIO} \
    --gradient_checkpointing \
    --weight_decay ${WEIGHT_DECAY} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    2>&1 | tee "${LOG_FILE}" \
    && python run_rag.py --rerank_model_path "${OUTPUT_DIR}"