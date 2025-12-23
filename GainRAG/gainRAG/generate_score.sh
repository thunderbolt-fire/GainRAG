#!/bin/bash

# 修改为可以在后台运行
nohup python -m llm_supervision.construct_hf --task TriviaQA > generate_score.log 2>&1 &

echo "任务已在后台运行，进程ID: $!"
echo "日志输出到 generate_score.log"