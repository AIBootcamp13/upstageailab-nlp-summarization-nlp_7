import re
import pandas as pd
import os
import sys
import readline
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

def org(x):
    x = x.strip().replace("\n", "")

    texts = re.split(r"(?=#person\d+#:)", x, flags=re.IGNORECASE)

    if texts and texts[0].strip() == "":
      texts = texts[1:]
    
    return texts


def input_with_prefill(prompt, text):
    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    try:
        return input(prompt)
    finally:
        readline.set_pre_input_hook()

def text_organize(df):
    organized_dialogues = []

    print("Organizing dialogues...")
    unorganized_dialogues = df["dialogue"].apply(org)

    print(f"Total dialogues to organize: {len(unorganized_dialogues)}")

    for dialogue in tqdm(unorganized_dialogues, desc="Organizing dialogues"):
        new_dialogue = []
        flag = False

        for line in dialogue:
            if not re.search(r"[.!?]$", line.strip()):
                flag = True
                print("\n❗ 종결 부호가 없는 문장이 감지되었습니다.")
                edited = input_with_prefill("👉 문장을 편집하세요 (Enter 시 그대로 유지): ", line).strip()
                line = edited if edited else line

            new_dialogue.append(line)

        # 종결 부호 감지된 문장이 있었던 경우, 전체 대화 다시 보여주기
        if flag:
            print("\n💬 현재 전체 대화 내용:")
            preview = "\n".join(new_dialogue)
            print(preview)

            print("\n🔧 전체 대화를 한 번에 수정하고 싶다면 아래에 입력하세요.")
            edited_all = input_with_prefill("👉 전체 편집 (Enter 시 그대로 유지): ", preview).strip()

            if edited_all:
                # 다시 줄 단위로 나눔
                new_dialogue = edited_all.splitlines()

        organized_dialogues.append(new_dialogue)

    # DataFrame 갱신
    new_dialogues = ["\n".join(dialogue) for dialogue in organized_dialogues]
    df = df.copy()
    df["dialogue"] = new_dialogues

    return df

if __name__ == "__main__":

  df = pd.read_csv(os.path.join(ROOT_DIR, "data", "train.csv"))
  df = text_organize(df)
  df.to_csv(os.path.join(ROOT_DIR, "data", "train_organized.csv"), index=False)