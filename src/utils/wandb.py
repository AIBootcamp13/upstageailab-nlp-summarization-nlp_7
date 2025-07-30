import wandb
import os
import time
from dotenv import load_dotenv

def wandb_init(config):
    env_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '.env'))
    load_dotenv(env_path)

    wandb.login(key=os.getenv('WANDB_API_KEY'))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_folder_name = config['general']['model_folder_name']
    wandb_name = f"{model_folder_name}_{timestamp}"

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        name=wandb_name,
    )

    #모델 체크포인트 저장
    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"