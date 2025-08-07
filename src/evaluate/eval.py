import os 
import sys
import torch
import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from rouge import Rouge

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(ROOT_DIR)

from src.utils.config import load_config

from src.model.bart import load_tokenizer_and_model_for_test
from src.preprocess.preprocess import Preprocess
from src.dataset.datamodule import prepare_eval_dataset


def eval(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config, device)

    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_eval_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'],shuffle=False)

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm.tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    rouge = Rouge()
    rouge_result = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "rouge_avg": []
    }
    for i in range(len(preprocessed_summary)):
        results = rouge.get_scores(preprocessed_summary[i], test_data['summary'][i])
        result = {key: value["f"] for key, value in results[0].items()}
        rouge_avg = (result["rouge-1"] + result["rouge-2"] + result["rouge-l"]) / 3
        rouge_result["rouge-1"].append(result["rouge-1"])
        rouge_result["rouge-2"].append(result["rouge-2"])
        rouge_result["rouge-l"].append(result["rouge-l"])
        rouge_result["rouge_avg"].append(rouge_avg)


    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "dialogue": test_data['dialogue'],
            "gold": test_data['summary'],
            "pred" : preprocessed_summary,
            "coverage_ratio": test_data['coverage_ratio'],
            "rouge-1": rouge_result["rouge-1"],
            "rouge-2": rouge_result["rouge-2"],
            "rouge-l": rouge_result["rouge-l"],
            "rouge_avg": rouge_result["rouge_avg"]
        }
    )

    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, f"{config['general']['model_folder_name']}_eval_output.csv"), index=False)

    return output


if __name__ == "__main__":
    config_adj, config = load_config()
    eval(config_adj)