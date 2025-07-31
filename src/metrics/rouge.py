import evaluate

def compute_rouge(config,tokenizer,pred):
    rouge = evaluate.load("rouge")
    preds, labels = pred

    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = decoded_labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.compute(predictions=replaced_predictions, references=replaced_labels, use_stemmer=True)

    # f1 기준 점수만 추출
    rouge_1 = results["rouge1"]
    rouge_2 = results["rouge2"]
    rouge_l = results["rougeL"]
    rouge_avg = (rouge_1 + rouge_2 + rouge_l) / 3

    
    return {
            "rouge-1": rouge_1,
            "rouge-2": rouge_2,
            "rouge-l": rouge_l,
            "rouge_avg": rouge_avg
        }
