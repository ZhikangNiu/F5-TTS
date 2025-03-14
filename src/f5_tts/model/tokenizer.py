from functools import lru_cache
import torch
from transformers import AutoTokenizer

class QwenTokenizer():
    def __init__(self, token_path, skip_special_tokens=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                '[breath]', '<strong>', '</strong>', '[noise]',
                '[laughter]', '[cough]', '[clucking]', '[accent]',
                '[quick_breath]',
                "<laughter>", "</laughter>",
                "[hissing]", "[sigh]", "[vocalized-noise]",
                "[lipsmack]", "[mn]"
            ]
        }
        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    token_path: str,
    skip_special_tokens: bool
) -> QwenTokenizer:
    return QwenTokenizer(token_path=token_path, skip_special_tokens=skip_special_tokens)