import os
import sys
import yaml
import copy

ROOT_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', ".."))
sys.path.append(ROOT_DIR)

def load_config(name="config"):
    config_path = os.path.join(ROOT_DIR, 'config', f'{name}.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 루트 기준으로 경로 변경
    # 원본 파일은 건드리지 않고 수정된 파일과 같이 반환
    config_adj = copy.deepcopy(config)
    config_adj['general']['data_path'] = os.path.join(ROOT_DIR, config['general']['data_path'])
    config_adj['general']['output_dir'] = os.path.join(ROOT_DIR, config['general']['output_dir'], config['general']['model_folder_name'])
    config_adj['inference']['result_path'] = os.path.join(ROOT_DIR, config['inference']['result_path'])
    return config_adj, config

def save_config(config, name="config"):
    config_path = os.path.join(ROOT_DIR, 'config', f'{name}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)