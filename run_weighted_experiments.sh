#!/bin/bash

# Array of 5 random seeds
seeds=(42 123 456 789 1024)

cd ./Weighted_Sampler
# Loop through each seed and run the command
for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_weighted_policy.py" \
        --config_path ./config/knee_acl_config.yaml \
        --seed "$seed"

    # Optional: add a small delay between runs
    sleep 1
done

for seed in "${seeds[@]}"; do
    echo "Running with seed: $seed"
    CUDA_VISIBLE_DEVICES=0 python "train_weighted_policy.py" \
        --config_path ./config/knee_cart_config.yaml \
        --seed "$seed"

    # Optional: add a small delay between runs
    sleep 1
done


echo "All weighted policy runs completed!"