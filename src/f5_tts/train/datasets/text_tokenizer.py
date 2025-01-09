#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
# Copyright    2024                            (authors: Zhikang Niu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
from typing import List, Pattern, Union

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator



SYMBOLS_MAPPING = {
    "：": ":",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "、": ",",
    "...": "…",
    "......": "…",
    "‘": "“",
    "’": "”",
    "（": "(",
    "）": ")",
    "《": "«",
    "》": "»",
    "<": "«",
    ">": "»",
    "【": "[",
    "】": "]",
    "—": "—",
    "～": "—",
    "~": "—",
    "「": '"',
    "」": '"',
    "/": ",",
    "\\": ",",
    "^":"'",
    "\"":"'",
    "·": ",",
    '�':",",
    # some tn
    "@":" at ",
    "&":" and ",
    "%":" percent "
    # default marks
    # "[": "'",
    # "]": "'",
    # "(": "'",
    # ")": "'",
    # "“": "'",
    # "”": "'",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)

class PhonemizeTextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(), # default_marks = ';:,.!?¡¿—…"«»“”(){}[]'
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE) 
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def clean_text(self,text):
        # Clean the text
        text = text.strip()
        text = regex.sub(r'\p{C}|\p{Z}', ' ', text) # ignore \n \t \v \f \r \u2028
        text = REPLACE_SYMBOL_REGEX.sub(lambda x: SYMBOLS_MAPPING[x.group()], text)
        return text
    
    def __call__(self, text, strip=True) -> List[List[str]]:
        # try:
        if isinstance(text, str):
            text = [self.clean_text(text)]
        elif isinstance(text, list):
            text = [self.clean_text(t) for t in text]
        else:
            print("Only support text_list input and str input")
        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]


def tokenize_text(tokenizer: PhonemizeTextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes  # k2symbols

if __name__ == "__main__":
    tt = PhonemizeTextTokenizer()
    text = "hello, how are you"
    # (Summary by Tom Weiss) «,p»'
    print(text)
    print(tokenize_text(tt,text))
    cmn_tokenizer = PhonemizeTextTokenizer(language="cmn",language_switch="remove-flags",with_stress=True)
    cmn_text = "你好"
    print(cmn_text)
    print(tokenize_text(cmn_tokenizer,cmn_text))
    # ja_text = "日本語にほんご本good"
    # print(ja_text)
    # ja_tokenizer = PhonemizeTextTokenizer(language="ja",language_switch="remove-flags",with_stress=False)
    # print(tokenize_text(ja_tokenizer,ja_text))