#!/bin/bash

# Uncomment the line below if you need to set CUDA memory allocation configurations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"

# Source the path configurations
. ./path.sh

# Setting CUDA device to use
export CUDA_VISIBLE_DEVICES=3

# Running the Python script with specified arguments for Stage 3 (Clustering)
python src/federated_main.py \
    --model=data2vec \
    --gpu=0 \
    --pretrain_name "./data2vec-audio-large-960h-local/" \
    --frac=1.0 \
    --num_users=5 \
    --global_ep=10 \
    --learning_rate=1e-5 \
    --num_lms 5 \
    --training_type 1 \
    --local_ep 10 \
    --epochs 10 \
    --N_Kmeans_update 10 \
    --FL_STAGE 1 \
    -model_out "./data2vec-audio-large-960h" \
    -model_in "./data2vec-audio-large-960h" \
    --dataset_path_root "./ADReSS-copy/datasets" \
    --Kmeans_model_path "./save/k_means_model" \
    --eval_mode 2 \
    --FL_type 1 \
    --mu 0.01 \
    --alpha 0.5 \
    --beta 0.5 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --use_soft_clustering \
    --fuzzy_m 2 \
    --membership_threshold 0.175 \
    --use_membership_weighted_avg \
    # --use_rl_fcm \
    # --FML_model 0 \
    # --CBFL \
    # --WeightedAvg \

# Handle SIGINT and SIGTERM for graceful termination of background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait