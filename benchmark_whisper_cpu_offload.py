import argparse
import csv
import subprocess as sp
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperConfig, WhisperProcessor

from modeling_whisper_cpu_offload import WhisperCPUOffloadForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a Whisper model with Flash Attention and Torch compile.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for the benchmark.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of batches over which to evaluate.",
    )
    parser.add_argument(
        "--generated_tokens",
        type=int,
        default=25,
        help="Number of tokens to generate for each example.",
    )
    parser.add_argument(
        "--output_csv_file",
        type=str,
        default="results.csv",
        help="Where to write the benchmark results.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default="tiny.en base.en small.en medium.en",
        help="Which checkpoints to evaluate.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=4,
        type=int,
        help="Number of workers for the PyTorch dataloader.",
    )

    args = parser.parse_args()
    return args


def get_gpu_memory():
    """Python equivalent of nvidia-smi, modified from https://stackoverflow.com/a/67722676"""
    def output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]
    command = "nvidia-smi --query-gpu=memory.used --format=csv"

    try:
        memory_use_info = output_to_list(sp.check_output(command.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    memory_use_values = [int(x.split()[0]) for x in memory_use_info]
    return memory_use_values


def main():
    args = parse_args()

    whisper_cls = WhisperCPUOffloadForConditionalGeneration

    # benchmark on 100 samples from the LS dataset
    librispeech = load_dataset("sanchit-gandhi/librispeech_asr_clean", split="train.100")
    librispeech = librispeech.select(range(args.batch_size * args.num_batches))

    # processors/tokenizers are the same for all models, so just load from tiny and preprocess once
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

    def preprocess(batch):
        batch["input_features"] = processor(
            batch["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features[0]
        return batch

    dataset_processed = librispeech.map(preprocess, remove_columns=librispeech.column_names)

    dataloader = DataLoader(
        dataset_processed.with_format("torch"),
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # dicts to store our results
    whisper_checkpoints = list(args.checkpoints.split(" "))
    decoder_layer_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
    runtime_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
    param_results = {checkpoint: [] for checkpoint in whisper_checkpoints}
    vram_results = {checkpoint: [] for checkpoint in whisper_checkpoints}

    # evaluate on the largest model first so that an OOM happens sooner rather than later
    for checkpoint in whisper_checkpoints:
        print(10 * "=", checkpoint, 10 * "=")
        checkpoint_id = f"openai/whisper-{checkpoint}"
        config = WhisperConfig.from_pretrained(checkpoint_id)

        if checkpoint == "large-v2":
            layer_increments = [1, 2, 4, 6, 8, 16, 32]
        elif checkpoint == "medium.en":
            layer_increments = [1, 2, 4, 6, 8, 16, 24]
        else:
            total_decoder_layers = config.decoder_layers
            layer_increments = np.arange(2, total_decoder_layers + 2, 2)
            layer_increments = np.insert(layer_increments, 0, 1)

        layer_increments = layer_increments[::-1]

        for idx, encoder_layers in enumerate(layer_increments):
            config.encoder_layers = int(encoder_layers)
            print("Encoder layers: ", encoder_layers)
            for decoder_layers in layer_increments[idx:]:
                print("Decoder layers: ", decoder_layers)
                config.decoder_layers = int(decoder_layers)
                model = whisper_cls(config)
                model.to("cuda").half()

                start = time.time()
                for batch in tqdm(dataloader):
                    input_features = batch["input_features"].to("cuda").half()
                    pred_ids = model.generate(
                        input_features, max_new_tokens=args.generated_tokens, min_new_tokens=args.generated_tokens
                    )
                runtime = time.time() - start

                decoder_layer_results[checkpoint].append(f"{int(encoder_layers)}-{int(decoder_layers)}")
                runtime_results[checkpoint].append(runtime)
                param_results[checkpoint].append(model.num_parameters() / 10**6)
                vram_results[checkpoint].append(get_gpu_memory()[0])

                del model
                torch.cuda.empty_cache()

    # Save the results
    compression_results = {}
    for checkpoint in param_results:
        original_params = param_results[checkpoint][0]
        compression_results[checkpoint] = [
            100 * (original_params - compressed_params) / original_params
            for compressed_params in param_results[checkpoint]
        ]

    # Save the results
    headers = ["Checkpoint", "Layers", "Params / M", "Compression / %", "VRAM / GB", "Runtime / s"]
    with open(args.output_csv_file, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the headers
        writer.writerow(headers)
        # write the data
        for key in decoder_layer_results:
            for i in range(len(decoder_layer_results[key])):
                prefix = key.replace(".en", "").replace("-v2", "") if i == 0 else ""
                data = [
                    prefix,
                    decoder_layer_results[key][i],
                    round(param_results[key][i], 1),
                    round(compression_results[key][i], 1),
                    round(vram_results[key][i] / 1000, 2),
                    round(runtime_results[key][i], 1),
                ]
                writer.writerow(data)
            writer.writerow([])


if __name__ == "__main__":
    main()
