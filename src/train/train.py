from transformers import Seq2SeqTrainer, EarlyStoppingCallback, DataCollatorForSeq2Seq
import wandb
import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', ".."))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

from src.train.seq2seqarg import load_seq2seqarg
from src.metrics.rouge import compute_rouge
from src.model.bart import load_tokenizer_and_model_for_train, load_tokenizer_and_model_for_test
from src.preprocess.preprocess import Preprocess
from src.dataset.datamodule import prepare_train_dataset
from src.utils.wandb import wandb_init
from src.utils.config import load_config, save_config


def train(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model, tokenizer = load_tokenizer_and_model_for_test(config, device)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)
    print(tokenizer.special_tokens_map)

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

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
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_rouge(config, tokenizer, pred),
        callbacks = [MyCallback]
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    trainer.train()

    wandb.finish()
    return trainer.state.best_model_checkpoint


if __name__ == "__main__":
    config_adj, config = load_config()
    best_model_checkpoint = train(config_adj)

    config['inference']['ckt_path'] = best_model_checkpoint
    save_config(config)
    print(best_model_checkpoint)
