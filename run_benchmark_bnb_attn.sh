#!/usr/bin/env bash

python ./benchmark_whisper_bnb_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --output_csv_file="bnb_large_bs.csv"

python ./benchmark_whisper_bnb_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --use_flash_attn \
    --output_csv_file="bnb_attn_large_bs.csv"
