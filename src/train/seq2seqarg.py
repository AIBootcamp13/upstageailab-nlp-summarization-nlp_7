from transformers import Seq2SeqTrainingArguments

def load_seq2seqarg(config):
    return Seq2SeqTrainingArguments(
                output_dir=config['general']['output_dir'], # model output directory
                **config['training']
    )