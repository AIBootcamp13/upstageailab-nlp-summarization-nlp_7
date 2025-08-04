from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

def load_tokenizer_and_model_for_train(config, device):

    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if config['tokenizer']['pad_token_as_eos']:
    #     tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.padding_side = config['tokenizer']['padding_side']

    # special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    # tokenizer.add_special_tokens(special_tokens_dict)

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
        attn_implementation="sdpa",
    )
    # generate_model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        target_modules=config['lora']['target_modules'],
        task_type=TaskType.CAUSAL_LM
    )
    generate_model = get_peft_model(generate_model, lora_config)
    

    return generate_model , tokenizer

def load_tokenizer_and_model_for_test(config, adapter_path=None):
    print('-'*10, 'Load tokenizer & model for inference', '-'*10)
    model_name = config['general']['model_name']
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if config['tokenizer']['pad_token_as_eos']:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # tokenizer.add_special_tokens({'additional_special_tokens': config['tokenizer']['special_tokens']})

    # bitsandbytes 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['bnb']['load_in_4bit'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config['bnb']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=config['bnb']['bnb_4bit_quant_type']
    )

    # base model 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    # base_model.resize_token_embeddings(len(tokenizer))

    # PEFT 어댑터 로드
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("No adapter path provided. Using base model only.")
        model = base_model

    model.eval()  # 추론용
    return model, tokenizer