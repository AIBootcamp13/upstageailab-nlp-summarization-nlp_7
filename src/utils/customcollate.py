from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence

def custom_causal_lm_collator(tokenizer):
    def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]

        # (1) pad input_ids and labels
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="right")
        labels = pad_sequence(labels, batch_first=True, padding_value=-100, padding_side="right")

        # (2) attention_mask 계산 시, 이제 input_ids/labels는 Tensor임!
        # pad_token_id와 -100 둘 다일 경우 → 0, 나머지는 1
        attention_mask = ((input_ids != tokenizer.pad_token_id) | (labels != -100)).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    return collate_fn

def custom_causal_lm_collator_eval(tokenizer):
    TEMP_PAD = -1
    def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]

        # (1) pad input_ids and labels
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=TEMP_PAD, padding_side="left")

        
        attention_mask = ((input_ids != TEMP_PAD)).long()

        input_ids[input_ids == TEMP_PAD] = tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dialogue": [x['dialogue'] for x in batch],
            "summary": [x['summary'] for x in batch]
        }

    return collate_fn


def custom_causal_lm_collator_infer(tokenizer):
    TEMP_PAD = -1
    def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]

        # (1) pad input_ids and labels
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=TEMP_PAD, padding_side="left")

        
        attention_mask = ((input_ids != TEMP_PAD)).long()

        input_ids[input_ids == TEMP_PAD] = tokenizer.pad_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "fname": [x['fname'] for x in batch]
        }

    return collate_fn