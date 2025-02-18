#!/bin/bash

# 设置环境变量
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 配置参数
CONFIG_FILE="test_pretrain_template.json"
ALIGNED_PROPORTION=1000

## 运行所有 (meta, fine) 组合
#for META in "" "--meta"; do
#    for FINE in "" "--fine"; do
#        # 构建实验标识符，将空字符串替换为"false"，"--meta"替换为"true"
#        META_ID=$([ "$META" = "--meta" ] && echo "true" || echo "false")
#        FINE_ID=$([ "$FINE" = "--fine" ] && echo "true" || echo "false")
#        LOG_FILE="experiment_meta_${META_ID}_fine_${FINE_ID}.log"
#
#        echo "Running experiment with meta=${META_ID}, fine=${FINE_ID}..."
#
#        # 运行main.py
#        python main.py \
#            --config $CONFIG_FILE \
#            --aligned_proportion $ALIGNED_PROPORTION \
#            $META $FINE | tee $LOG_FILE
#    done
#done

# 运行main_img.py的实验
#for META in "" "--meta"; do
#    for FINE in "" "--fine"; do
#        META_ID=$([ "$META" = "--meta" ] && echo "true" || echo "false")
#        FINE_ID=$([ "$FINE" = "--fine" ] && echo "true" || echo "false")
#        LOG_FILE="experiment_meta_${META_ID}_fine_${FINE_ID}_img.log"  # 添加img标识
#
#        echo "Running experiment with meta=${META_ID}, fine=${FINE_ID}..."
#
#        python main_img.py \
#            --config $CONFIG_FILE \
#            --aligned_proportion $ALIGNED_PROPORTION \
#            $META $FINE | tee $LOG_FILE
#    done
#done

# 运行main_fmnist.py的实验
for META in "" "--meta"; do
    for FINE in "" "--fine"; do
        META_ID=$([ "$META" = "--meta" ] && echo "true" || echo "false")
        FINE_ID=$([ "$FINE" = "--fine" ] && echo "true" || echo "false")
        LOG_FILE="experiment_meta_${META_ID}_fine_${FINE_ID}_fmnist.log"  # 添加fmnist标识

        echo "Running experiment with meta=${META_ID}, fine=${FINE_ID}..."

        python main_fmnist.py \
            --config $CONFIG_FILE \
            --aligned_proportion $ALIGNED_PROPORTION \
            $META $FINE | tee $LOG_FILE
    done
done

echo "All experiments completed!"

## 分析日志文件
#echo "================= Experiment Summary ================="
#for META in "false" "true"; do
#    for FINE in "false" "true"; do
#        LOG_FILE="experiment_meta_${META}_fine_${FINE}.log"
#        echo "Results for meta=$META, fine=$FINE:"
#
#        # 统计日志中的关键信息（例如训练损失、测试精度）
#        grep -E "Loss|Accuracy|Evaluation" $LOG_FILE | tail -10
#        echo "----------------------------------------------------"
#    done
#done
