#!/bin/bash

# Array of 5 random seeds
seeds=(42 123 456 789 1024)

cd ./Classifier
# Loop through each seed and run the command
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_classifier.py" \
        --config_path ./config/knee_acl_config.yaml \
        --seed "$seed" \
        --initial_accelerations 20 \
        --final_accelerations   4
    # Optional: add a small delay between runs
    sleep 1
done

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_classifier.py" \
        --config_path ./config/knee_cart_config.yaml \
        --seed "$seed" \
        --initial_accelerations 20 \
        --final_accelerations   4
    # Optional: add a small delay between runs
    sleep 1
done

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_classifier.py" \
        --config_path ./config/knee_acl_degree_config.yaml \
        --seed "$seed" \
        --initial_accelerations 20 \
        --final_accelerations   4
    # Optional: add a small delay between runs
    sleep 1
done

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_classifier.py" \
        --config_path ./config/knee_cart_degree_config.yaml \
        --seed "$seed" \
        --initial_accelerations 20 \
        --final_accelerations   4
    # Optional: add a small delay between runs
    sleep 1
done


echo "All classifier runs completed!"