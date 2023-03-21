#!/usr/bin/env bash

python ./benchmark_whisper_attn.py \
    --output_csv_file="low_bs_large.csv"

python ./benchmark_whisper_attn.py \
    --use_flash_attn \
    --output_csv_file="flash_low_bs_large.csv"
