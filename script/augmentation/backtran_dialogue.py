from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd
import os
import sys
import torch
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)



def nllb_batch_translate(texts, src_lang, tgt_lang, batch_size=4, tokenizer=None, model=None, device=None):
    results = []
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=1024
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

def backtranslation(df, config, batch_size=4):
        
    special_tokens = config['tokenizer']['special_tokens']
    # NLLB 모델 로드
    MODEL_NAME = "facebook/nllb-200-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    backtranslated_dialogues = []

    # 원본 dialogue 전처리: 개행 제거 (줄마다 #PersonX#: 있는 경우만)
    dialogue_texts = [dialogue.strip().replace("\n", " ") for dialogue in df['dialogue']]

    # 역번역 수행
    try:
        en_batch = nllb_batch_translate(dialogue_texts, src_lang="kor_Hang", tgt_lang="eng_Latn", batch_size=batch_size, tokenizer=tokenizer, model=model, device=device)
        ko_back_batch = nllb_batch_translate(en_batch, src_lang="eng_Latn", tgt_lang="kor_Hang", batch_size=batch_size, tokenizer=tokenizer, model=model, device=device)
    except Exception as e:
        print(f"[번역 실패]: {e}")
        ko_back_batch = [f"[번역실패] {d}" for d in dialogue_texts]

    backtranslated_dialogues = ko_back_batch
    df = df.copy()
    df['dialogue'] = backtranslated_dialogues
    return df

if __name__ == "__main__":
    config_path = os.path.join(ROOT_DIR, "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_organized.csv"))
    df = backtranslation(df.head(10), config, batch_size=4)
    df.to_csv(os.path.join(ROOT_DIR, "data", "train_organized_backtranslated.csv"), index=False)
