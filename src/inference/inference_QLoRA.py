import os
import sys
import torch
import tqdm

from rouge import Rouge
from torch.utils.data import DataLoader
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(ROOT_DIR)

from src.model.LLM_QLoRA import load_tokenizer_and_model_for_test
from src.preprocess.preprocess import Preprocess
from src.dataset.datamodule_QLoRA import prepare_test_dataset
from src.utils.config import load_config
from src.utils.customcollate import custom_causal_lm_collator_infer

def inference(config):
    model, tokenizer = load_tokenizer_and_model_for_test(config, adapter_path=config['inference']['ckt_path'])
    print(model.device)

    preprocessor = Preprocess(None, None)
    test_dataset = prepare_test_dataset(config, preprocessor, config['general']['data_path'], tokenizer)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False,
        collate_fn=custom_causal_lm_collator_infer(tokenizer)
        # collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    )

    predictions = []
    fnames = []
    for batch in tqdm.tqdm(test_dataloader, desc="Inferring"):
        fname = batch["fname"]
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

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
        fnames.extend(fname)


    ouputs = [pred.split("### Output:")[-1].strip() for pred in predictions]

    df = pd.DataFrame({'fname': fnames, 'summary': ouputs})
    df.to_csv(os.path.join(ROOT_DIR, 'outputs', 'prediction', 'qlora_result.csv'), index=False)

            

if __name__ == "__main__":
    adj_config, _ = load_config("config_QLoRA")
    inference(adj_config)