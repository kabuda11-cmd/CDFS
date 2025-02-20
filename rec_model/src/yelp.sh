#!/bin/bash

# 定义替换的部分，可以根据需要修改这个列表
replace_list=("insert1" "insert2" "insert3")  # 你可以在这里定义更多的替换值

# 初始化模型索引和种子值
model_idx=1
seed=10

# 遍历replace_list中的每个值
for insert_value in "${replace_list[@]}"; do
    # 构建新的模型路径
    model_cl_path="./model_cl_pretrain/CoSeRec-Yelp-${insert_value}.pt"
    
    # 执行命令
    python main.py --output_dir output_Yelp/ --data_name Yelp --model_cl_path $model_cl_path --model_idx $model_idx --seed $seed
    
    # 更新model_idx和seed
    model_idx=$((model_idx + 1))
    seed=$((seed + 1))
done