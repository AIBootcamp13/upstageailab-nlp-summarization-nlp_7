# 📌 Dialogue Summarization Baseline

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
|:----------------------------------------------------------------:|:----------------------------------------------------------------:|:----------------------------------------------------------------:|:----------------------------------------------------------------:|:----------------------------------------------------------------:|
| [박패캠](https://github.com/UpstageAILab) | [이패캠](https://github.com/UpstageAILab) | [최패캠](https://github.com/UpstageAILab) | [김패캠](https://github.com/UpstageAILab) | [오패캠](https://github.com/UpstageAILab) |
| 팀장, 총괄 기획 | 데이터 분석, EDA | 모델링, 파인튜닝, 성능 개선 | 실험 자동화, 스크립트 개발 | 실험 관리, 리포트 정리 및 발표자료 제작 |

---

## 0. Overview

### Environment

- Python 3.10  
- CUDA 11.8  
- PyTorch 2.7.1  
- Huggingface Transformers 4.41.0  
- Datasets 4.0.0  
- Bitsandbytes 0.46.1  
- Accelerate 0.29.3  

### Requirements

```bash
pip install -r requirements.txt
```
 `requirements.txt`:
```
pandas==2.1.4
numpy==1.23.5
wandb==0.16.1
tqdm==4.66.1
pytorch_lightning==2.1.2
transformers[torch]>=4.41.0
rouge==1.0.1
jupyter==1.0.0
jupyterlab==4.0.9
python-dotenv==1.0.1
gradio==5.38.2
seaborn
nltk
konlpy
scikit-learn
umap-learn
hdbscan
sentencepiece==0.2.0
peft==0.16.0
datasets==4.0.0
accelerate>=0.24.0
bitsandbytes>=0.46.1
```

---

## 1. Competition Info

### Overview

- 제공된 500개의 대화문을 요약하는 NLP 태스크  
- 다양한 주제의 대화문을 정확하고 간결하게 요약하는 모델 개발이 목표  
- 추론 결과는 500개의 요약문을 `.csv`로 제출

### Timeline

- July 25, 2025 - 대회 시작  
- August 6, 2025 - 제출 마감  

---

## 2. Components

### Directory Structure

```
.
├── config/
├── data/
├── notebooks/
│   └── KJB/
├── script/ # gradio를 활용한 결과 분석 script 등
├── src/
│   ├── dataset/           # 데이터 전처리, 로더, 토크나이저
│   │   └── preprocess.py
│   │   └── loader.py
│
│   ├── model/             # 모델 아키텍처 및 로딩
│   │   └── base_model.py
│   │   └── lora_wrapper.py
│   │   └── peft_loader.py
│
│   ├── train/             # 학습 로직
│   │   └── train_qlora.py
│   │   └── trainer.py
│
│   ├── evaluation/        # 평가 및 메트릭
│   │   └── evaluator.py
│   │   └── metrics.py
│
│   ├── inference/         # 추론 및 생성
│   │   └── infer.py
│   │   └── generator.py
│
│   ├── util/              # 공통 유틸 함수
│   │   └── collator.py
│   │   └── logger.py
│   │   └── config_loader.py
│
│   └── main.py            # 실행 진입점 (예: argparse 기반 전체 파이프라인)
├── .env.template
├── .gitignore
├── README.md
├── requirements.txt

```

---

## 3. Data Description

### Dataset Overview

- `train.csv`: 12457개 대화문 + 요약  
- `dev.csv`: 500개 대화문 + 요약
- 'train.csv': 500개 대화문  
- 대화 형식: turn 기반 (`#Person1#: ...`, `#Person2#: ...`)

### EDA

- 평균 발화 수: 7.8  
- 평균 요약 길이: 23.1 토큰  
- Coverage ratio 분석 결과: 일부 요약이 지나치게 상세하거나 중복된 표현 포함

### Data Processing

- 특수 문자 제거, 불필요한 공백 정리  
- 템플릿 삽입  
  ```
  ### Instruction: 다음 대화를 요약해줘.
  ### Input:
  (대화)
  ### Output:
  ```
- Truncation 및 max_length 지정

---

## 4. Modeling

### Model Description

- **Base**: `Upstage/SOLAR-10.7B-Instruct`  
- **Fine-tuning**: QLoRA 적용 (4-bit 양자화)  
- **Tokenizer**: pad_token = eos_token = `</s>`  
- **LoRA target modules**: `q`, `k`, `v`, `o`, `gate`, `up`, `down`

### Modeling Process

1. 프롬프트 구성:
    ```
    ### Instruction: 다음 대화를 요약해줘.
    ### Input:
    #Person1#: ...
    #Person2#: ...
    ### Output:
    ```

2. 학습:
   - Trainer 기반 QLoRA fine-tuning (3 epoch)  
   - `Seq2SeqTrainer`로 ROUGE-L, ROUGE-2 평가  

3. 추론:
   - Beam Search 기반 생성  
   - `.csv` 형태로 결과 저장  

---

## 5. Result

### Leaderboard

- **ROUGE-L**: 42.3  
- **순위**: 🥈 2위  

> 📸 리더보드 캡처 삽입 예정

### Presentation

- [📎 발표자료 (PDF)](https://drive.google.com/...)

---

## etc

### Meeting Log

- [📒 회의록 (Notion)](https://www.notion.so/...)

### Reference

- [Huggingface Transformers](https://huggingface.co/docs)
- [Upstage/SOLAR 모델 카드](https://huggingface.co/Upstage/SOLAR-10.7B-Instruct)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
