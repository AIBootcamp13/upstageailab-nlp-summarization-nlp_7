from transformers import TrainerCallback
from transformers.trainer_utils import PredictionOutput
import torch

class SamplePredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, interval_steps=8):
        self.tokenizer = tokenizer
        self.interval_steps = interval_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        # interval 마다만 출력
        if state.global_step % self.interval_steps != 0 or state.global_step == 0:
            return

        trainer = kwargs['model']
        inputs = kwargs['inputs']

        # input_ids 하나만 가져오기
        input_ids = inputs["input_ids"][0].unsqueeze(0).to(trainer.device)
        label_ids = inputs["labels"][0].unsqueeze(0).to(trainer.device)

        # 예측 (단순 greedy로 빠르게)
        output_ids = trainer.generate(
            input_ids=input_ids,
            max_new_tokens=64,
            do_sample=False
        )

        pred_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        label_text = self.tokenizer.decode(label_ids[0], skip_special_tokens=True)

        print("=" * 50)
        print(f"[Step {state.global_step}] Prediction:")
        print(pred_text)
        print(f"Label:")
        print(label_text)
        print("=" * 50)
