from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
libritts_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS"

def read_libritts_jsonl(libritts_path):
    texts = []
    paths = list(Path(libritts_path).rglob("*.normalized.txt"))
    for path in tqdm(paths, desc="Reading texts"):
        with open(path, "r") as f:
            text = f.read().strip()
            texts.append(text)
    return texts


def set_trainer(vocab_size, min_frequency):
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=["<|unk|>"]
    )
    return trainer

def train_tokenizer(texts, save_path, vocab_size, min_frequency):
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = set_trainer(vocab_size, min_frequency)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)

texts = read_libritts_jsonl(libritts_path)
print(f"Total texts: {len(texts)}")
for vocab_size in [256, 1024, 4096]:
    save_path = f"./libritts/tokenizer_{vocab_size}_min_frequency_2.json"
    if not Path(save_path).exists():
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    train_tokenizer(texts, save_path, vocab_size, min_frequency=2)



    