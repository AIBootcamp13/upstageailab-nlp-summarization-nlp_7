from transformers import Trainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
import wandb
import os
import sys
import torch
from peft import PeftModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', ".."))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

from src.train.seq2seqarg import load_seq2seqarg
from src.metrics.rouge import compute_rouge
from src.model.LLM_QLoRA import load_tokenizer_and_model_for_train
from src.preprocess.preprocess import Preprocess
from src.dataset.datamodule_QLoRA import prepare_train_dataset
from src.utils.wandb import wandb_init
from src.utils.config import load_config, save_config

# def compute(pred, tokenizer, config):
#     print('-'*10, 'compute', '-'*10,)
#     preds, labels = pred
#     labels[labels == -100] = tokenizer.pad_token_id
#     decoded_labels = tokenizer.batch_decode(labels)

#     print(decoded_labels)
#     return 0


def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model, tokenizer = load_tokenizer_and_model_for_train(config, device)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

    tokenizer.padding_side = 'right'

    preprocessor = Preprocess(None, None)
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer, 1000, 250)

    print('-'*10, 'Make training arguments', '-'*10,)

    if not os.path.exists(os.path.join(ROOT_DIR, config['general']['output_dir'])):
        os.makedirs(os.path.join(ROOT_DIR, config['general']['output_dir']))

    if not os.path.exists(os.path.join(ROOT_DIR, config['general']['output_dir'], config['general']['model_folder_name'])):
        os.makedirs(os.path.join(ROOT_DIR, config['general']['output_dir'], config['general']['model_folder_name']))

    # set training args
    training_args = load_seq2seqarg(config)

    #wandb 불러오기
    wandb_init(config)

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['early_stopping']['early_stopping_patience'],
        early_stopping_threshold=config['early_stopping']['early_stopping_threshold']
    )

    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    # Trainer 클래스를 정의합니다.
    trainer = Trainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        tokenizer=tokenizer,
        # compute_metrics = lambda pred: compute(pred, tokenizer, config)
        # callbacks = [MyCallback],
    )
    print(f"Using device: {trainer.args.device}")
    print('-'*10, 'Make trainer complete', '-'*10,)

    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint

    generate_model.save_pretrained(best_ckpt, save_adapter=True)

    wandb.finish()
    return best_ckpt


if __name__ == "__main__":
    print(torch.cuda.is_bf16_supported())
    config_adj, config = load_config(name="config_QLoRA")
    best_model_checkpoint = train(config_adj)

    config['inference']['ckt_path'] = best_model_checkpoint
    save_config(config, name="config_QLoRA")
    print(best_model_checkpoint)
