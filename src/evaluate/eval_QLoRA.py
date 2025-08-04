import os
import sys
import torch
import tqdm
from torch.utils.data import DataLoader
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(ROOT_DIR)

from src.model.LLM_QLoRA import load_tokenizer_and_model_for_test
from src.preprocess.preprocess import Preprocess
from src.dataset.datamodule_QLoRA import prepare_eval_dataset
from transformers import DataCollatorForSeq2Seq
import evaluate as hf_evaluate
from src.utils.config import load_config

def evaluate(config):
    model, tokenizer = load_tokenizer_and_model_for_test(config, adapter_path=config['inference']['ckt_path'])
    print(model.device)

    preprocessor = Preprocess(None, None)
    eval_dataset = prepare_eval_dataset(config, preprocessor, config['general']['data_path'], tokenizer, 3)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
        # collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    )

    predictions = []
    references = []
    dialogues = []

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):

        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,             # or True
                # num_beams=4,                 # ✔ beam search 추천
                # no_repeat_ngram_size=3,     # ✔ 반복 방지
                # repetition_penalty=1.1,     # ✔ 반복 방지
                # length_penalty=0.8,         # ✔ 요약 짧게 유도
                # early_stopping=True,
                # eos_token_id=tokenizer.eos_token_id,
                # pad_token_id=tokenizer.pad_token_id
            )

        # 디코딩 (batch size > 1일 수 있음)
        decoded_preds = tokenizer.batch_decode(outputs)
        decoded_preds = [pred.replace(tokenizer.eos_token, "") for pred in decoded_preds]
        predictions.extend(decoded_preds)

    df = pd.DataFrame({'predictions': predictions})
    df.to_csv(os.path.join(config['general']['data_path'], 'eval_results.csv'), index=False)

    # ROUGE 계산
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("Evaluation Results:")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

    return result, predictions, references
            

if __name__ == "__main__":
    adj_config, _ = load_config("config_QLoRA")
    evaluate(adj_config)