# Emilia Dataset: https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07
# if use updated new version, i.e. WebDataset, feel free to modify / draft your own script

# generate audio text map for Emilia ZH & EN
# evaluate for vocab size

import sys, os
sys.path.append(os.getcwd())

from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter

from f5_tts.model.utils import repetition_found

from text_tokenizer import tokenize_text,PhonemizeTextTokenizer
from write_lang_id import LANGUAGES,LANG2IPA
import pykakasi
kks = pykakasi.kakasi()

out_zh = {
    "ZH_B00041_S06226",
    "ZH_B00042_S09204",
    "ZH_B00065_S09430",
    "ZH_B00065_S09431",
    "ZH_B00066_S09327",
    "ZH_B00066_S09328",
}
zh_filters = ["い", "て"]
# seems synthesized audios, or heavily code-switched
out_en = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}
en_filters = ["ا", "い", "て"]

LANGUAGES = {'af': 'Afrikaans', 
             'am': 'Amharic', 
             'an': 'Aragonese', 
             'ar': 'Arabic',
             'as': 'Assamese',
             'az': 'Azerbaijani',
             'ba': 'Bashkir',
             'bg': 'Bulgarian', 
             'bn': 'Bengali',
             'bs': 'Bosnian',
             'ca': 'Catalan',
             'cs': 'Czech',
             'cy': 'Welsh',
             'da': 'Danish',
             'de': 'German',
             'el': 'Greek', 
             'en': 'English', 
             'eo': 'Esperanto', 
             'es': 'Spanish', 
             'eu': 'Basque', 
             'fa': 'Persian',
             'fi': 'Finnish',
             'fr': 'French',
             'ga-IE': 'Irish',
             'gn': 'Guarani',
             'gu-IN': 'Gujarati',
             'he': 'Hebrew',
             'hi': 'Hindi', 
             'hr': 'Croatian',
             'ht': 'Haitian',
             'hu': 'Hungarian',
             'hy-AM': 'Armenian',
             'hyw': 'Armenian Western',
             'ia': 'Interlingua',
             'id': 'Indonesian',
             'is': 'Icelandic', 
             'it': 'Italian',
             'ja': 'Japanese',
             'jbo': 'Lojban',
             'ka': 'Georgian', 
             'kk': 'Kazakh', 
             'kn': 'Kannada', 
             'ko': 'Korean', 
             'ky': 'Kyrgyz',
             'lb': 'Luxembourgish',
             'lt': 'Lithuanian', 
             'ltg': 'Latgalian',
             'lv': 'Latvian',
             'mk': 'Macedonian',
             'ml': 'Malayalam', 
             'mni': 'Meetei Lon',
             'mr': 'Marathi',
             'ms': 'Malay',
             'mt': 'Maltese',
             'my': 'Burmese',
             'nb-NO': 'Norwegian Bokmål',
             'ne-NP': 'Nepali', 
             'nl': 'Dutch',
             'om': 'Afaan Oromo',
             'or': 'Odia',
             'pa-IN': 'Punjabi', 
             'pl': 'Polish',
             'pt': 'Portuguese',
             'quc': "K'iche'",
             'ru': 'Russian',
             'sd': 'Sindhi', 
             'si': 'Sinhala',
             'sk': 'Slovak',
             'sq': 'Albanian',
             'sr': 'Serbian',
             'sv-SE': 'Swedish',
             'sw': 'Swahili',        
             'ta': 'Tamil',  
             'th': 'Thai',
             'tk': 'Turkmen',
             'tn': 'Setswana',
             'tt': 'Tatar',
             'ug': 'Uyghur',
             'uk': 'Ukrainian',
             'ur': 'Urdu',
             'uz': 'Uzbek', 
             'vi': 'Vietnamese',
             'yue': 'Cantonese',
             'zh-CN': 'Chinese (China)',
             'zh-HK': 'Chinese (Hong Kong)',
             'zh-TW': 'Chinese (Taiwan)'
             }

LANG2IPA = {
    "zh-CN" : "zh",
    "zh-HK" : "zh",
    "zh-TW" : "zh",
    "ga-IE" : "ga",
    "gu-IN" : "gu",
    'hy-AM': "hy",
    'nb-NO' : "nb",
    'ne-NP' : "ne",
    'pa-IN': "pa",
    "sv-SE": "sv",
    "fr" : "fr"
}

def deal_with_audio_dir(audio_dir,text_tokenizer):
    audio_jsonl = audio_dir.with_suffix(".jsonl")
    sub_result, durations = [], []
    vocab_set = set()
    bad_case_zh = 0
    bad_case_en = 0
    with open(audio_jsonl, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"{audio_jsonl.stem}"):
            obj = json.loads(line)
            text = obj["text"]
            lang_id = obj["language"]
            duration = obj["duration"]
            if lang_id == "zh":
                if obj["wav"].split("/")[1] in out_zh or any(f in text for f in zh_filters) or repetition_found(text):
                    bad_case_zh += 1
                    continue
            elif lang_id == "en":
                if obj["wav"].split("/")[1] in out_en or any(f in text for f in en_filters) or repetition_found(text, length=4):
                    bad_case_en += 1
                    continue
            elif lang_id == "ja":
                results = kks.convert(text)
                text = "".join(result["hira"] for result in results) # -> hiragana
            phonemes = tokenize_text(text=text,tokenizer=text_tokenizer)
            try:
                sub_result.append({
                    "audio_path": str(audio_dir.parent / obj["wav"]),
                    "language":lang_id, 
                    "text": phonemes[0],
                    "duration": duration
                })
                durations.append(duration)
                vocab_set.update(phonemes[0])
            except:
                print(f"text: {text}")
                print(f"phonemes: {phonemes}")

    return sub_result, durations, vocab_set, bad_case_zh, bad_case_en


def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    total_bad_case_zh = 0
    total_bad_case_en = 0
    
    phonemizer_dict = {
        "ZH" : "cmn",
        "EN" : "en-us",
        "DE" : "de",
        "FR" : "fr-fr",
        'JA': "ja",
        'KO' : "ko",
    }
    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []
    
    for lang in langs:
        dataset_path = Path(os.path.join(dataset_dir, lang))
        text_tokenizer = PhonemizeTextTokenizer(
            language=phonemizer_dict[lang],
            language_switch="remove-flags",
            with_stress=False
        )
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir,text_tokenizer))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for futures in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, bad_case_zh, bad_case_en = futures.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        total_bad_case_zh += bad_case_zh
        total_bad_case_en += bad_case_en
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nSaving to {output_dir} ...")
    # dataset = Dataset.from_dict({"audio_path": audio_path_list, "text": text_list, "duration": duration_list})  # oom
    # dataset.save_to_disk(f"data/{dataset_name}/raw", max_shard_size="2GB")
    with ArrowWriter(path=f"{output_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc=f"Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{output_dir}/duration.json", 'w', encoding='utf-8') as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    with open(f"{output_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")
    
    langs = []
    for key in LANGUAGES.keys():
        if key in LANG2IPA.keys():
            langs.append(LANG2IPA[key])
        else:
            langs.append(key)

    with open("lang.txt","w") as f:
        for line in set(langs):
            f.write(f"{line}\n")


    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list)/3600:.2f} hours")
    if "ZH" in langs: print(f"Bad zh transcription case: {total_bad_case_zh}")
    if "EN" in langs: print(f"Bad en transcription case: {total_bad_case_en}\n")


if __name__ == "__main__":

    max_workers = 64
    tokenizer = "ipa"
    langs = ["ZH", "EN", "DE", "JA", "KO","FR"]
    # dataset_dir = "<SOME_PATH>/Emilia_Dataset/raw"
    dataset_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/datasets/Emilia-Dataset"
    dataset_name = f"Emilia_{'_'.join(langs)}_{tokenizer}_lang_id"
    output_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}\n")

    main()

    # Emilia               ZH & EN
    # samples count       37837916   (after removal)
    # pinyin vocab size       2543   (polyphone)
    # total duration      95281.87   (hours)
    # bad zh asr cnt        230435   (samples)
    # bad eh asr cnt         37217   (samples)

    # vocab size may be slightly different due to jieba tokenizer and pypinyin (e.g. way of polyphoneme)
    # please be careful if using pretrained model, make sure the vocab.txt is same
