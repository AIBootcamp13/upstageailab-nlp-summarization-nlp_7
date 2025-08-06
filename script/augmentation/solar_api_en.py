import pandas as pd
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

load_dotenv()

UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')

def build_prompt(dialogue: str):
    system_prompt = (
        "당신은 영어 대화를 한국어로 번역하는 전문 번역가입니다.\n\n"
        "주어진 영어 대화를 한국어로 자연스럽고 문맥에 맞게 번역하세요.\n\n"
        "규칙:\n"
        "1. 각 발화를 단어 그대로 직역하지 말고, 문맥에 맞게 자연스럽게 의역하세요.\n"
        "2. 발화는 turn 형식(예: #Person1#: ..., #Person2#: ...)을 그대로 유지해야 합니다.\n"
        "3. 발화의 개수는 원문과 동일하게 유지하세요.\n"
        "4. 인물 이름은 번역하지 마세요. 예를 들어 Mr. Smith, Dr. Kim, Jessica, John 등 영어 이름은 모두 원문 그대로 출력하세요.\n"
        "5. 번역된 대화만 출력하세요. 그 외의 설명이나 요약은 포함하지 마세요."
    )

    user_prompt = f"영어 대화:\n{dialogue.strip()}\n\n번역된 대화:"
    assistant_prompt = (
      "예시: \n"
      "입력: \n"
      "#Person1#: Hello, Mr. Smith.\n"
      "출력: \n"
      "#Person1#: 안녕하세요, Mr. Smith.\n"
      "입력: \n"
      "#Person1#: Hello, Dr. Kim."
      "출력: \n"
      "#Person1#: 안녕하세요, Dr. Kim."
      "입력: \n"
      "#Person1#: Hello, Jessica."
      "출력: \n"
      "#Person1#: 안녕하세요, Jessica.\n"
      "입력: \n"
      "#Person1#: I'm Malik."
      "출력: \n"
      "#Person1#: 나는 Malik이야.\n"
      "입력: \n"
      "#Person1#: I'm Nicholas. Nice to meet you."
      "출력: \n"
      "#Person1#: 나는 Nicholas야. 반가워.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prompt}
    ]

def augment_data(df, start_idx=0, end_idx=100):
    client = OpenAI(
        api_key=os.getenv('UPSTAGE_API_KEY'),
        base_url="https://api.upstage.ai/v1/solar"
    )

    result = {
        "dialogue": [],
        "summary": [],
        "topic": [],
        "fname": []
    }

    df = df.iloc[start_idx:end_idx]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dialogue = row['dialogue']
        summary = row['summary']
        topic = row['topic']
        fname = row['fname']

        messages = build_prompt(dialogue)

        stream = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=messages,
            temperature=1.2,           # ↑ 창의성 증가
            top_p=0.8,                 # 확률 질량 샘플링 (nucleus sampling)
            # max_tokens=1024,
            frequency_penalty=0.2,
            stream=False
        )
        output = stream.choices[0].message.content
        if output is not None:
            result["dialogue"].append(output)
            result["summary"].append(summary)
            result["topic"].append(topic)
            result["fname"].append(fname)
        else:
            continue
    aug_df = pd.DataFrame(result)
    aug_df.to_csv(os.path.join(ROOT_DIR, 'data', f'train_solar_aug_{start_idx+1000}_{end_idx+1000}.csv'), index=False)
    
    

if __name__ == "__main__":
  df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_backtranslated_sentence_1000_4000.csv'))
  augment_data(df, start_idx=0, end_idx=1000)