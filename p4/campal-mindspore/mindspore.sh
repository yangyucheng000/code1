#!/bin/bash

# 定义存储脚本文件名的数组
scripts=(
    "./sh/Entroy_normal.sh"
    "./sh/LC_normal.sh"
)

# 循环遍历并执行脚本
for script in "${scripts[@]}"; do
    echo "Running $script ..."
    bash "$script"
    echo "Finished running $script."
done

echo "All scripts executed successfully."
