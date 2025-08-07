from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
import pandas as pd
import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

# NLLB 모델 로드
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def nllb_batch_translate(texts, src_lang, tgt_lang, batch_size=8):
    results = []
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=5,
            no_repeat_ngram_size=2,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

def backtranslation(df, range_start=0, range_end=None, batch_size=8):
    backtranslated_dialogues = []

    for dialogue in tqdm(df['dialogue'][range_start:range_end], desc="Backtranslation"):
        x = dialogue.strip().replace("\n", "")
        lines = re.split(r"(?=#person\d+#:)", x, flags=re.IGNORECASE)
        if lines and lines[0].strip() == "":
            lines = lines[1:]

        tag_text_pairs = []
        texts = []

        for line in lines:
            line = line.strip()
            if line == "":
                continue
            try:
                tag, text = line.split(":", 1)
                tag = tag.strip()
                text = text.strip()
                tag_text_pairs.append(tag)
                texts.append(text)
            except ValueError:
                continue  # 잘못된 포맷 무시

        try:
            en_batch = nllb_batch_translate(texts, src_lang="eng_Latn", tgt_lang="kor_Hang", batch_size=batch_size)
            # ko_back_batch = nllb_batch_translate(en_batch, src_lang="eng_Latn", tgt_lang="kor_Hang", batch_size=batch_size)
        except Exception as e:
            en_batch = [f"[번역실패] {text}" for text in texts]

        # 태그 복원
        backtranslated_lines = [
            f"{tag}: {bt}" for tag, bt in zip(tag_text_pairs, en_batch)
        ]

        result = "\n".join(backtranslated_lines)
        backtranslated_dialogues.append(result)

    df = df.copy()
    df['dialogue'] = backtranslated_dialogues
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_backtranslated_sentence_0_1000.csv"))
    range_start = 0
    range_end = 5
    df = df.iloc[range_start:range_end]
    df = backtranslation(df, batch_size=8)
    df.to_csv(os.path.join(ROOT_DIR, "data", f"train_backtranslated_sentence_{range_start}_{range_end}_ko.csv"), index=False)
