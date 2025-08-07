import pandas as pd
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

load_dotenv()

UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')

def build_prompt(dialogue: str, topic: str):
    system_prompt = (
        "You are an expert in dialogue paraphrasing. Your task is to rewrite the given dialogue "
        "in a different way while preserving the original meaning.\n\n"
        "Rules:\n"
        "1. You must keep the dialogue in a turn-based format (e.g., #Person1#: ... #Person2#: ...).\n"
        "2. Paraphrase the dialogue naturally and fluently in Korean.\n"
        "3. The number of turns does not have to be the same as the original, but the conversational flow must be preserved.\n"
        "4. Keep English names (e.g., Mr. Smith, Dr. Kim) unchanged.\n"
        "5. Use the given topic to better understand the context and choose more appropriate wording.\n"
        "6. Only output the paraphrased dialogue. Do not include any other text.\n\n"
        "You will receive two inputs: a topic and a dialogue. Output only the paraphrased dialogue in Korean."
    )

    user_prompt = f"Dialogue:\n{dialogue}\n\nTopic:\n{topic}\n\n재작성 대화:"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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

    for idx, row in df.iterrows():
        dialogue = row['dialogue']
        summary = row['summary']
        topic = row['topic']
        fname = row['fname']

        messages = build_prompt(dialogue, topic)

        stream = client.chat.completions.create(
            model="solar-pro2-250710",
            messages=messages,
            temperature=1.2,           # ↑ 창의성 증가
            top_p=0.8,                 # 확률 질량 샘플링 (nucleus sampling)
            # max_tokens=1024,
            frequency_penalty=0.2,
            stream=False
        )
        output = stream.choices[0].message.content
        print(output)

        if output is not None:
            result["dialogue"].append(output)
            result["summary"].append(summary)
            result["topic"].append(topic)
            result["fname"].append(fname)
        else:
            continue
    aug_df = pd.DataFrame(result)
    aug_df.to_csv(os.path.join(ROOT_DIR, 'data', f'train_solar_aug_{start_idx}_{end_idx}.csv'), index=False)
    
    

if __name__ == "__main__":
  df = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'train_organized.csv'))
  augment_data(df, start_idx=0, end_idx=5)