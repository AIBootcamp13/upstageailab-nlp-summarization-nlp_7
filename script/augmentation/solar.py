import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import sys
from tqdm import tqdm
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

SYSTEM_PROMPT_TEXT = """
너는 한국어 대화를 자연스럽게 바꿔주는 어시스턴트야.

규칙:
- 발화 수는 절대 바꾸지 마.
- 화자 태그 (예: "#Person1#:")는 그대로 유지해.
- 영어 이름 (예: "Mr. Smith", "Dr. Hawkins")은 번역하거나 바꾸지 마.
- 자연스럽고 다양한 한국어 표현으로 바꿔.
- 오직 바꾼 대화문만 출력해.

예시:
입력:
#Person1#: 지금 뭐 해?
#Person2#: 그냥 쉬고 있어.

출력:
#Person1#: 뭐 하고 있어?
#Person2#: 그냥 좀 쉬는 중이야
"""

def build_prompt(dialogue: str) -> str:
    return f"""{SYSTEM_PROMPT_TEXT.strip()}

입력:
{dialogue.strip()}

출력:"""

def extract_assistant_response(output: str) -> str:
    # SOLAR는 특별한 토큰을 출력하지 않음 → 그대로 사용
    return output.strip()

def chat_with_solar(df):
    dialogues = df['dialogue']

    model_name = "Upstage/SOLAR-10.7B-Instruct-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    generated_dialogues = []
    for dialogue in tqdm(dialogues, desc="SOLAR Generation"):
        prompt = build_prompt(dialogue)
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = extract_assistant_response(outputs[0]["generated_text"])
        generated_dialogues.append(generated)

    df['dialogue'] = generated_dialogues
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_organized.csv"))
    df = chat_with_solar(df.head(5))
    df.to_csv(os.path.join(ROOT_DIR, "data", "train_organized_solar.csv"), index=False)
