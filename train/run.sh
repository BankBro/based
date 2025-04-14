#!/bin/bash

# 定义GPU利用率阈值（例如10%）
THRESHOLD=0.9

# 定义要执行的命令
COMMAND_TO_RUN="HF_HUB_ENABLE_HF_TRANSFER=1 HF_ENDPOINT=https://hf-mirror.com HYDRA_FULL_ERROR=1 python run.py experiment=example/04-14-1-based-vq-360m trainer.devices=1"

# 检查GPU利用率的函数
check_gpu_utilization() {
    # 获取GPU利用率
    UTILIZATION=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{used=$1; total=$2; printf "%.2f\n", (used/total)}')
    if (( $(echo "$UTILIZATION < $THRESHOLD" | bc -l) )); then
        echo "GPU utilization ${UTILIZATION} is below the threshold, executing the command..."
        return 0
    fi
    return 1
}

while true; do
    check_gpu_utilization
    if [ $? -eq 0 ]; then
        date
        eval $COMMAND_TO_RUN
        break
    fi
    sleep 30
done



