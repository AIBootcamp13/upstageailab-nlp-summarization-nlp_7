from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch

def load_tokenizer_and_model_for_train(config, device):

    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if config['tokenizer']['pad_token_as_eos']:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['bnb']['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config['bnb']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=config['bnb']['bnb_4bit_quant_type']
    )

    generate_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    generate_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=config['lora']['r'],
        alpha=config['lora']['alpha'],
        dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules'],
        task_type=TaskType.CAUSAL_LM
    )
    generate_model = get_peft_model(generate_model, lora_config)
    

    return generate_model , tokenizer

def load_tokenizer_and_model_for_test(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)

    model_name = config['general']['model_name']
    ckt_path = config['inference']['ckt_path']
    print('-'*10, f'Model Name : {model_name}', '-'*10,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    print('-'*10, 'Load tokenizer & model complete', '-'*10,)

    return generate_model , tokenizer