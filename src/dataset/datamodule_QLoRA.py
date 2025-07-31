import os
import sys
from datasets import Dataset

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(ROOT_DIR)

from src.dataset.dataset_bart import DatasetForTrain, DatasetForVal, DatasetForInference



def format_sample(example, tokenizer, config):
    return {
        "input_ids": tokenizer(
            f"### Instruction: 다음 대화를 요약해줘.\n\n### Input:\n{example['dialogue']}",
            max_length=config['tokenizer']['max_source_length'],
            padding="max_length",
            truncation=True,
            return_tensors=None
        )["input_ids"][0],
        "labels": tokenizer(
            example['summary'],
            max_length=config['tokenizer']['max_target_length'],
            padding="max_length",
            truncation=True,
            return_tensors=None
        )["input_ids"][0]
    }

def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'train.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

   
    print('-'*10, 'Load data complete', '-'*10,)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    train_dataset = train_dataset.map(lambda x: format_sample(x, tokenizer, config), remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(lambda x: format_sample(x, tokenizer, config), remove_columns=val_dataset.column_names)

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_dataset, val_dataset


def prepare_test_dataset(config, preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path, is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , _ = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset