from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
import pandas as pd
import os
import sys
import torch

import spacy

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

# NER 모델 로딩 (spaCy)
nlp = spacy.load("en_core_web_sm")

# NLLB 모델 로드
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


TITLES = {"Mr.", "Ms.", "Mrs.", "Dr.", "Prof."}

def mark_names(texts):
    docs = list(nlp.pipe(texts, batch_size=16))
    marked_texts = texts.copy()
    name_map = {}

    for i, doc in enumerate(docs):
        already_replaced = set()
        j = 0
        while j < len(doc):
            token = doc[j]
            # PERSON 시작
            if token.ent_iob_ == "B" and token.ent_type_ == "PERSON":
                name_tokens = [token.text]
                k = j + 1
                # 뒤에 I-ENT 따라붙는 이름 계속 추가
                while k < len(doc) and doc[k].ent_iob_ == "I":
                    name_tokens.append(doc[k].text)
                    k += 1

                # 앞에 타이틀 있는지 확인
                if j > 0 and doc[j - 1].text in TITLES:
                    name_tokens.insert(0, doc[j - 1].text)
                    j = j - 1  # 타이틀도 포함시켜서 처리해야 하므로 j 감소

                full_name = " ".join(name_tokens)

                # 중복 방지
                if full_name in already_replaced:
                    j = k
                    continue

                placeholder = f"NAME{i}_{j}"
                name_map[placeholder] = full_name
                marked_texts[i] = re.sub(re.escape(full_name), f"##{placeholder}##", marked_texts[i])
                already_replaced.add(full_name)
                j = k  # 다음 토큰으로 이동
            else:
                j += 1

    return marked_texts, name_map

def restore_names(text_list, name_map):
    restored_texts = []

    for text in text_list:
        restored_text = text
        for placeholder, name in name_map.items():
            restored_text = restored_text.replace(placeholder, name)
        restored_texts.append(restored_text)

    return restored_texts

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

def backtranslation(df, batch_size=8):
    backtranslated_dialogues = []

    for dialogue in tqdm(df['dialogue'], desc="Backtranslation"):
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

        marked_text, name_map = mark_names(texts)
        print(f"marked_text: {marked_text}")
        print(f"name_map: {name_map}")

        try:
            en_batch = nllb_batch_translate(marked_text, src_lang="eng_Latn", tgt_lang="kor_Hang", batch_size=batch_size)
            # ko_back_batch = nllb_batch_translate(en_batch, src_lang="eng_Latn", tgt_lang="kor_Hang", batch_size=batch_size)
        except Exception as e:
            en_batch = [f"[번역실패] {text}" for text in texts]
        en_batch = restore_names(en_batch, name_map)

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
    range_end = 15
    df = df.iloc[range_start:range_end]
    df = backtranslation(df, batch_size=8)
    df.to_csv(os.path.join(ROOT_DIR, "data", f"train_backtranslated_sentence_{range_start}_{range_end}_ko.csv"), index=False)
