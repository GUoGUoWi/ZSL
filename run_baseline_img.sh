#!/bin/bash

# 定义要运行的 aligned_proportion 值数组
proportions=(10 200 800 1200 3000)
WAIT_TIME=60  # 等待时间（秒）
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_img.py --aligned_proportion $prop --use_global_zero --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_img.py --aligned_proportion $prop  --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done

#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_fmnist_ensemble.py --aligned_proportion $prop --use_global_zero
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_fmnist_ensemble.py --aligned_proportion $prop
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_fmnist_ensemble.py --aligned_proportion $prop --use_global_zero --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_fmnist_ensemble.py --aligned_proportion $prop --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done

for prop in "${proportions[@]}"
do
    echo "Running main.py with aligned_proportion = $prop"
    python main_baseline_img.py --aligned_proportion $prop --use_global_zero
    echo "Completed run with aligned_proportion = $prop"

    # 如果不是最后一个运行，则等待
    if [ $prop != ${proportions[-1]} ]; then
        echo "Waiting for $WAIT_TIME seconds before next run..."
        sleep $WAIT_TIME
    fi

    echo "----------------------------------------"
done
echo "All runs completed!"

for prop in "${proportions[@]}"
do
    echo "Running main.py with aligned_proportion = $prop"
    python  main_baseline_img.py --aligned_proportion $prop
    echo "Completed run with aligned_proportion = $prop"

    # 如果不是最后一个运行，则等待
    if [ $prop != ${proportions[-1]} ]; then
        echo "Waiting for $WAIT_TIME seconds before next run..."
        sleep $WAIT_TIME
    fi

    echo "----------------------------------------"
done



for prop in "${proportions[@]}"
do
    echo "Running main.py with aligned_proportion = $prop"
    python main_baseline_img.py --aligned_proportion $prop --use_global_zero --sup
    echo "Completed run with aligned_proportion = $prop"

    # 如果不是最后一个运行，则等待
    if [ $prop != ${proportions[-1]} ]; then
        echo "Waiting for $WAIT_TIME seconds before next run..."
        sleep $WAIT_TIME
    fi

    echo "----------------------------------------"
done
echo "All runs completed!"

for prop in "${proportions[@]}"
do
    echo "Running main.py with aligned_proportion = $prop"
    python  main_baseline_img.py --aligned_proportion $prop --sup
    echo "Completed run with aligned_proportion = $prop"

    # 如果不是最后一个运行，则等待
    if [ $prop != ${proportions[-1]} ]; then
        echo "Waiting for $WAIT_TIME seconds before next run..."
        sleep $WAIT_TIME
    fi

    echo "----------------------------------------"
done

#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_ensemble.py --aligned_proportion $prop --use_global_zero --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_ensemble.py --aligned_proportion $prop --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done




#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_img_ensemble.py --aligned_proportion $prop --use_global_zero
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_img_ensemble.py --aligned_proportion $prop
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python main_baseline_img_ensemble.py --aligned_proportion $prop --use_global_zero --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done
#echo "All runs completed!"
#
#for prop in "${proportions[@]}"
#do
#    echo "Running main.py with aligned_proportion = $prop"
#    python  main_baseline_img_ensemble.py --aligned_proportion $prop --sup
#    echo "Completed run with aligned_proportion = $prop"
#
#    # 如果不是最后一个运行，则等待
#    if [ $prop != ${proportions[-1]} ]; then
#        echo "Waiting for $WAIT_TIME seconds before next run..."
#        sleep $WAIT_TIME
#    fi
#
#    echo "----------------------------------------"
#done

echo "All runs completed!"