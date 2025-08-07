import torch
from transformers import pipeline
import pandas as pd
import os
import sys
from tqdm import tqdm
import re
ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)


# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

SYSTEM_PROMPT = {
  "role": "system",
  "content": """
  You are a Korean dialogue rewriter. Your job is to paraphrase naturally in Korean while preserving structure and key elements.

  Rules:
  - Do not change the number of utterances.
  - Keep speaker tags (e.g., "#Person1#:") exactly as is.
  - Keep English names (e.g., "Mr. Smith", "Dr. Hawkins") untouched.
  - Only output the paraphrased dialogue.

  Example:
  Input:
  #Person1#: 지금 뭐 해?
  #Person2#: 그냥 쉬고 있어.

  Output:
  #Person1#: 뭐 하고 있어?
  #Person2#: 그냥 좀 쉬는 중이야
  """
}

def extract_assistant_response(generated_text: str) -> str:
    match = re.search(r"<\|assistant\|>\n(.*)", generated_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def chat_with_zephyr(df):
    dialouges = df['dialogue']
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")

    generated_dialogues = []
    for dialouge in tqdm(dialouges, desc="Zephyr Generation"):
        messages = [
            SYSTEM_PROMPT,
            {"role": "user", "content": dialouge}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_dialogues.append(extract_assistant_response(outputs[0]["generated_text"]))

    df['dialogue'] = generated_dialogues
    return df

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train_organized.csv"))
    df = chat_with_zephyr(df.head(5))
    df.to_csv(os.path.join(ROOT_DIR, "data", "train_organized_zephyr.csv"), index=False)