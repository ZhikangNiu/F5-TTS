import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from tokenizers import Tokenizer
import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

def set_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    token_lengths = []
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue
        if tokenizer_type == "bpe":
            split_text = bpe_tokenizer.encode(text).tokens # 查看切分后的文本
            token = bpe_tokenizer.encode(text).ids
        else:
            token = text
        sub_result.append({"audio_path": str(line), "text": token, "duration": duration})
        # sub_result.append({"audio_path": str(line), "text": token, "split_text": split_text, "duration": duration})
        durations.append(duration)
        vocab_set.update(list(token))
        token_lengths.append(len(token))
    return sub_result, durations, vocab_set, token_lengths


def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    all_token_lengths = []

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, token_lengths = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        all_token_lengths.extend(token_lengths)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for i,line in tqdm(enumerate(result), desc="Writing to raw.arrow ..."):
            if i < 10:
                print(line)
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(str(vocab) + "\n")

    # 计算并输出token长度统计信息
    avg_token_length = sum(all_token_lengths) / len(all_token_lengths)
    max_token_length = max(all_token_lengths)
    min_token_length = min(all_token_lengths)

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    print(f"For {dataset_name}, average token length: {avg_token_length:.2f}")
    print(f"For {dataset_name}, max token length: {max_token_length}")
    print(f"For {dataset_name}, min token length: {min_token_length}")


if __name__ == "__main__":
    max_workers = 36
    
    tokenizer_type = "bpe"  # "pinyin" | "char" | "bpe"
    vocab_size = 4096
    bpe_tokenizer = set_tokenizer(f"/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxie-25019/f5-prefix/bpe_config/tokenizer_{vocab_size}_min_frequency_2.json")
    SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{vocab_size}_{tokenizer_type}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
