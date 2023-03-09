#!/usr/bin/env bash

python ./benchmark_whisper_attn.py \
    --output_csv_file="original_low_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --output_csv_file="original_large_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --generated_tokens="256" \
    --output_csv_file="original_large_seq_len.csv"

python ./benchmark_whisper_attn.py \
    --use_flash_attn \
    --output_csv_file="flash_low_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --use_flash_attn \
    --output_csv_file="flash_large_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --generated_tokens="256" \
    --use_flash_attn \
    --output_csv_file="flash_large_seq_len.csv"

python ./benchmark_whisper_attn.py \
    --use_torch_compile \
    --output_csv_file="compile_low_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --use_torch_compile \
    --output_csv_file="compile_large_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --generated_tokens="256" \
    --use_torch_compile \
    --output_csv_file="compile_large_seq_len.csv"

python ./benchmark_whisper_attn.py \
    --use_flash_attn \
    --use_torch_compile \
    --output_csv_file="flash_compile_low_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --use_flash_attn \
    --use_torch_compile \
    --output_csv_file="flash_compile_large_bs.csv"

python ./benchmark_whisper_attn.py \
    --batch_size="32" \
    --num_batches="10" \
    --generated_tokens="256" \
    --use_flash_attn \
    --use_torch_compile \
    --output_csv_file="flash_compile_large_seq_len.csv"
