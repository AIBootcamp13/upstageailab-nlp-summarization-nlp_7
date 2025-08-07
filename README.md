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

- 문장 길이 분포 분석: dialogue와 summary의 길이 분석
 <img width="453" height="292" alt="image" src="https://github.com/user-attachments/assets/93e79264-a39f-4a28-af9e-8cefd9832d11" />
- coverage 분석: 요약문의 단어 중에 **원문에 존재하는 단어를 사용한 비율(Coverage)** 확인
 <img width="376" height="316" alt="image" src="https://github.com/user-attachments/assets/d743d452-e5e2-4da0-a60c-691c8e72f9dd" />

- coverage에 따른 요약 난이도 분석: Coverage ratio를 5개의 영역으로 binning하여 각 영역에서 모델 예측 후 rouge score 확인
<img width="388" height="339" alt="image" src="https://github.com/user-attachments/assets/a85a86a1-93b5-451e-9d26-4c0cdd096479" />

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

### Data Augumentation

- 소량의 라벨링 데이터를 극복하고 모델 일반화 능력 향상
- 구어체 표현의 다양성 확보 및 문체 일관성 유지

#### 사용한 기법: Upstage Solar API (Solar LLM)
- Dialogue Paraphrasing
	- 표현 다양화를 위한 Dialogue 재작성
- Summary Rewrite:
	- 동일 요약 대상에 대한 다양한 요약 생성

####  파라미터 설정

```
temperature: 1.1       # 다양성 확보 
top_p: 0.9             # 의미 보존 
frequency_penalty: 0.1 # 중복 억제
```
#### Few-shot Prompting 적용
- Solar API의 instruction 구조에 맞춰 "user → assistant" 포맷 구성
- 요약 문체 및 구조를 일관되게 유지하며 SFT 친화적 데이터 확보

---

## 4. Modeling

### Model Description
총 2가지 모델로 실험 진행

### Baseline Model : `KoBART`
- 모델 구조: Encoder-Decoder
- 특징:
	- 대화체의 기본 문법 및 구조 학습
	- HuggingFace 기반 fine-tuning 진행
- 한계점:
	- 긴 dialogue의 핵심 문맥을 잡지 못하고 요약 정확도(ROUGE) 낮음
	- 인코딩 과정에서 의미 손실 발생 → Context 파악 미흡

####도메인 적응 사전학습 (TAPT)
- 도입 배경
	- 대화체 특유 어휘/구문 구조가 사전학습된 모델에 적게 반영됨
	- 대화체에 특화된 encoder를 획득하고 downstream 성능 향상
- 방식
	- KoBART 모델의 encoder에 대해 Masked Language Modeling(MLM) 기반 비지도 학습
	- 이후 요약 태스크를 위한 fine-tuning 수행
- 기대 효과
	- 구어체 문맥 파악력 향상
	- 소량의 레이블만으로도 높은 성능 유지



### Baseline Model : `Upstage/SOLAR-10.7B-Instruct`
- **모델 구조**: Decoder-only LLM (GPT 계열)
- **특징**:
  - 다국어 instruction tuning 기반 모델
  - Upstage에서 공개한 LLM으로, `### Instruction: ... ### Input: ... ### Output:` 포맷에 강함
  - HuggingFace에서 `transformers` + `QLoRA`로 효율적 파인튜닝 가능
- **한계점**:
  - 거대한 파라미터로 학습에 필요한 자원이 매우 많다
  - 추론 시간이 오래 걸린다

---

#### Instruction 기반 미세조정 (SFT with QLoRA)
- **도입 배경**  
  - 기존 kobart 모델의 요약 결과를 보았을 때, dialogue의 핵심 문맥을 파악하지 못하는 경우가 자주 발생  
  - LLM 모델 자체를 downstream task에 학습시킨다면 핵심 문맥을 잘 파악하면서도 기존의 요약 형식에 맞게 예측 가능
  - QLoRA를 통해 메모리 효율을 확보하면서도 대형 모델을 미세조정 가능
  - 비슷한 파라미터를 가진 타 모델 대비 월등한 성능

- **방식**
  - SOLAR 10.7B 모델에 대해 `QLoRA` 기반 SFT 수행  
  - Prompt 템플릿 사용:
    ```
    ### Instruction: 다음 대화를 요약해줘.
    ### Input:
    #Person1#: ...
    #Person2#: ...
    ### Output:
    (요약)
    ```
  - LoRA 적용 대상: `q`, `k`, `v`, `o`, `gate`, `down`, `up` 등 핵심 block 포함 -> full fine-tuning과 비슷한 효과를 줌

- **기대 효과**
  - instruction 포맷 이해력 극대화 → 명시적 요약 명령에 강한 반응  
  - 대용량 파라미터를 활용해 정밀한 문맥 파악 가능  
  - 기존 KoBART 대비 ROUGE 향상 및 발화 간 관계 인식력 증가 

### Modeling Process

#### SOLAR + QLoRA 학습
1. 학습 프롬프트 구성(right padding):
    ```
    ### Instruction: 다음 대화를 요약해줘.
    ### Input:
    #Person1#: ...
    #Person2#: ...
    ### Output:
    {summary}
    ```

2. 학습:
   - Trainer 기반 QLoRA fine-tuning (1 epoch)  
   - summary 부분만 labeling해서 다른 프롬프트에 대해서는 loss 계산 못하게 막음
   - gradient_checkpoint, prepare_model_for_kbit_training, use_cache=False 옵션 등을 통해 gpu 메모리 최대한 확보
   - bs=2
   - lora_r:8, lora_alpha:32, dropout:0.05  

3. 추론:
   - grid 기반 생성
   - bs=2
   - 결과 문장 split 후 summary 부분만 저장할 수 있도록 post process  
   - `.csv` 형태로 결과 저장  

---

## 5. Result

### Leaderboard
- **순위**:  1위  

<p align="center">
  <img src="https://github.com/user-attachments/assets/fbcb6795-de9b-40b9-9241-14c9f4b276bd" width="45%" />
  <img src="https://github.com/user-attachments/assets/ac7af015-48a0-473b-9992-11b82c361bec" width="45%" />
</p>

### Presentation

- [📎 발표자료 (PDF)](https://drive.google.com/file/d/1KCp7ExnV50lV6QPp5LCabN_6-64wZWKT/view?usp=sharing)

### Reference

- [Huggingface Transformers](https://huggingface.co/docs)
- [Upstage/SOLAR 모델 카드]([https://huggingface.co/Upstage/SOLAR-10.7B-Instruct](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0))
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
