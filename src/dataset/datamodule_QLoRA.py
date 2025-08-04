import os
import sys
import torch
from datasets import Dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(ROOT_DIR)

from src.utils.sampling import stratified_sample


def format_sample(example, tokenizer, config):
    max_length = config['tokenizer']['max_length']               # 예: 1024
    max_summary_length = config['tokenizer']['max_summary_length'] - 1  # 예: 128
    max_prompt_length = max_length - max_summary_length - 2          # 예: 896

    bos = tokenizer.bos_token_id  # 예: <s>
    eos = tokenizer.eos_token_id  # 예: </s>
    pad = tokenizer.pad_token_id
    # 1. Prompt 토크나이즈 (고정 길이 이하로 자름)
    prompt = f"### Instruction: 다음 대화를 요약해줘.\n### Input:\n{example['dialogue']}\n### Output:\n"
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False
    )["input_ids"]

    # 2. Summary 토크나이즈 (128 토큰 이하로 제한)
    summary_ids = tokenizer(
        example["summary"],
        truncation=True,
        max_length=max_summary_length,
        add_special_tokens=False
    )["input_ids"]

    # 3. Concat
    input_ids = [bos] + prompt_ids + summary_ids + [eos]
    attention_mask = [1] * len(input_ids)

    # 4. Padding (right padding 기준)
    padding_len = max_length - len(input_ids)
    input_ids = input_ids + [pad] * padding_len
    attention_mask = attention_mask + [0] * padding_len

    # 5. Labels: prompt + padding은 -100, summary만 정답
    labels = [-100] * (len(prompt_ids) + 1) + summary_ids + [eos] + [-100] * padding_len
    # labels += [-100] * (max_length - len(labels))  # 오른쪽에 남은 padding

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(labels) == max_length

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


def format_sample_eval(example, tokenizer, config):
    prompt = f"### Instruction: 다음 대화를 요약해줘.\n### Input:\n{example['dialogue']}\n### Output:\n"
    
    model_inputs = tokenizer(
        prompt,
        max_length=config['inference']['input_max_length'],
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # 중요!!
    )
    
    return {
        "input_ids": model_inputs["input_ids"].squeeze(0),
        "attention_mask": model_inputs["attention_mask"].squeeze(0),
    }

def prepare_train_dataset(config, preprocessor, data_path, tokenizer, sample_size=None, sample_size_val=None):
    train_file_path = os.path.join(data_path,'train_organized_eda.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    if sample_size:
        train_data = stratified_sample(train_data, 'coverage_ratio', sample_size, 4)
    if sample_size_val:
        val_data = val_data.sample(sample_size_val)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"].iloc[0]}')
    print(f'train_label:\n {train_data["summary"].iloc[0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"].iloc[0]}')
    print(f'val_label:\n {val_data["summary"].iloc[0]}')

   
    print('-'*10, 'Load data complete', '-'*10,)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    train_dataset = train_dataset.map(lambda x: format_sample(x, tokenizer, config), remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(lambda x: format_sample(x, tokenizer, config), remove_columns=val_dataset.column_names)

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_dataset, val_dataset

def prepare_eval_dataset(config, preprocessor, data_path, tokenizer, sample_size=None):
    val_file_path = os.path.join(data_path, 'dev.csv')
    val_data = preprocessor.make_set_as_df(val_file_path)

    if sample_size:
        val_data = val_data.sample(sample_size)  

    dataset = Dataset.from_pandas(val_data)
    gen_dataset = dataset.map(lambda x: format_sample_eval(x, tokenizer, config), remove_columns=dataset.column_names)
    return gen_dataset


# def prepare_test_dataset(config, preprocessor, tokenizer):

#     test_file_path = os.path.join(config['general']['data_path'],'test.csv')

#     test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
#     test_id = test_data['fname']

#     print('-'*150)
#     print(f'test_data:\n{test_data["dialogue"][0]}')
#     print('-'*150)

#     encoder_input_test , _ = preprocessor.make_input(test_data,is_test=True)
#     print('-'*10, 'Load data complete', '-'*10,)

#     test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
#                     add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)

#     test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
#     print('-'*10, 'Make dataset complete', '-'*10,)

#     return test_data, test_encoder_inputs_dataset