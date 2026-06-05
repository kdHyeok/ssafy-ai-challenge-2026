# Notebooks

이 디렉토리는 VQA 대회 실험 흐름을 설명하는 대표 노트북과 실험 문서를 포함합니다. baseline, 모델 변경, EDA, 32B LoRA 학습, 선택적 재추론, soft ensemble, detection/TTA 확장 실험 순서로 구성했습니다.

## 대표 파일

| 파일 | 역할 | 핵심 내용 |
|---|---|---|
| `Baseline_0.70279.ipynb` | 초기 baseline | `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline. 384 고정 해상도, train 200개 샘플, 제출 CSV 생성까지 end-to-end 흐름 확인 |
| `model-change-experiment.md` | 모델 변경 실험 기록 | baseline 코드에서 모델 카드 기준으로 MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B를 실행한 과정과 점수 정리 |
| `train_0.9333.ipynb` | 32B 학습 | `Qwen/Qwen3-VL-32B-Instruct` + Unsloth/LoRA 학습. Qwen3-VL-8B 채택 이후 32B로 확장하고, pseudo-label, augmentation, a/b/c/d 확률 저장을 함께 적용한 실험 |
| `score_0.94402.ipynb` | 선택적 Thinking 재추론 | 여러 제출 사이에서 답이 갈리는 저합의 샘플을 `Qwen/Qwen3-VL-32B-Thinking`으로 재추론하는 후처리 실험 |
| `ensemble_0.94363.ipynb` | soft ensemble | Qwen3-VL-32B 추론 후 저신뢰 샘플에 Qwen3.5-27B를 적용하고, a/b/c/d 확률을 가중 결합하는 실험 |
| `experimental/EDA.ipynb` | EDA | dev 응답 불일치, 질문 유형, pseudo-label, count/material 문제 분석 |
| `experimental/detection_tta_0.93102.ipynb` | 후반 확장 실험 | 공개 detector 기반 crop 생성, 원본+crop 입력, TTA, VQA 확률 결합 실험 |

## 실험 흐름

### 1. Baseline 확보

대표 파일: `Baseline_0.70279.ipynb`

- 모델: `Qwen/Qwen2.5-VL-3B-Instruct`
- 설정: 4bit LoRA, `IMAGE_SIZE=384`, LoRA `r=8`, `lora_alpha=16`, train 200개 샘플
- 역할: 데이터 로드, 이미지 전처리, 프롬프트 구성, fine-tuning, submission 생성 확인
- 결과: 0.70279

### 2. 모델 변경 실험

대표 파일: `model-change-experiment.md`

| 모델 | 기록 점수 | 판단 |
|---|---:|---|
| `openbmb/MiniCPM-V-4_5` | 0.78341 | baseline보다 높았지만 Qwen3-VL-8B보다 낮음 |
| `google/gemma-4-E4B-it` | 0.82815 | 대회 당일 출시 모델이라 레퍼런스가 부족했고 Qwen3-VL-8B보다 낮음 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | 가장 높아 Qwen3-VL-32B 확장의 기준으로 채택 |

이 단계에서는 baseline 코드에서 모델만 바꾸는 것을 우선했습니다. Hugging Face 모델 카드의 processor, model class, chat template, dtype, trust_remote_code 여부를 확인해 실행 가능성을 맞췄습니다.

### 3. EDA와 데이터 진단

대표 파일: `experimental/EDA.ipynb`

- train/test/dev 구조 확인
- dev 응답 다수결과 agreement rate 분석
- 질문 유형 분류: count, material, color, text/brand
- 정답 분포와 선택지 편향 확인
- image resize/padding/crop이 count/material 문제와 어떤 관련이 있을지 후속 실험 후보로 정리

EDA는 직접적인 성능 향상 claim이 아니라, 이후 32B 학습, confidence routing, Thinking 재추론, detection crop 실험의 문제의식으로 해석합니다.

### 4. 학습 파라미터 조정

수업 메모와 직접 비교 실험에서 확인한 축은 다음과 같습니다.

- 초기 loss가 class 수 기준 기대값과 비슷한지 확인
- 작은 샘플로 빠르게 overfit 되는지 확인
- `learning_rate=3e-5`, `constant_with_warmup`, `cosine`, `label_smoothing=0.1`, `adamw_torch` 비교
- MiniCPM-V-4_5에서 특정 설정이 답변을 a로 고정시키는 문제 확인
- train loss와 eval loss를 분리해 scheduler와 learning rate를 판단

32B 학습에서는 다음 설정을 사용했습니다.

| 항목 | 설정 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-32B-Instruct` |
| learning rate | `1e-4` |
| scheduler | `cosine` |
| batch size | `24` |
| gradient accumulation | `1` |
| epochs | `5` |
| LoRA r | `32` |
| LoRA alpha | `32` |
| weight decay | `0.01` |

후속 비교에서는 `learning_rate=5e-5`, batch size 16, `lora_alpha=16`, warmup 비율 조정도 검토했습니다.

### 5. Qwen3-VL-32B LoRA 학습

대표 파일: `train_0.9333.ipynb`

- 모델: `Qwen/Qwen3-VL-32B-Instruct`
- 프레임워크: Unsloth, TRL SFTTrainer, PEFT LoRA
- 학습 전략: train 전체와 dev pseudo-label 후보 활용, augmentation, sequence length 점검
- 조정한 주요 축: epoch, learning rate, batch size, LoRA rank/alpha, pseudo-label agreement 기준
- a/b/c/d별 softmax 확률 저장으로 선택적 재추론과 soft ensemble의 비교 기준 마련

### 6. 선택적 Thinking 재추론

대표 파일: `score_0.94402.ipynb`

- 모델: `unsloth/Qwen3-VL-32B-Thinking-unsloth-bnb-4bit`
- 여러 제출이 같은 답을 고른 쉬운 샘플과 답이 갈리는 저합의 샘플 분리
- answer disagreement 또는 낮은 confidence가 있는 샘플에만 Thinking 모델 적용
- 대표 산출물: `main-sub-think-2.csv`
- 결과: 0.94402

Thinking은 모든 문제에 일괄 적용한 전략이 아니라, 불확실한 샘플을 대상으로 한 후처리 전략입니다.

### 7. Soft ensemble

대표 파일: `ensemble_0.94363.ipynb`

- Model A: Qwen3-VL-32B
- Model B: Qwen3.5-27B
- 방식: a/b/c/d softmax 확률을 가중 합산
- 일부 실험에서는 confidence가 낮은 샘플에만 보조 모델 호출
- 결과: 0.94363

이 전략은 hard voting이 아니라 확률 기반 soft ensemble입니다. 최고점 0.94402보다 낮았기 때문에, 최종 핵심 성과보다는 후반 비교 실험으로 정리합니다.

### 8. Detection/TTA 확장

대표 파일: `experimental/detection_tta_0.93102.ipynb`

- Grounding DINO, Florence-2 공개 detector로 test 이미지의 관심 객체 후보 crop 생성
- detector는 별도 추가학습 없이 추론 보조 모듈로 사용
- 원본 이미지와 crop 이미지를 VQA 추론 입력에 함께 활용
- Qwen3-VL-32B LoRA와 Qwen3.5-27B의 원본/좌우반전 TTA 확률 결합
- 결과: 0.93102

이 실험은 VLM이 재활용품 객체를 세밀하게 인식하지 못하는 문제를 보완하려는 후반 확장 실험입니다. detector 결과를 pseudo-label이나 fine-tuning label로 사용하지 않았고, 추론 시 crop/TTA 보조 입력으로만 사용했습니다.

### 9. Falcon-Perception 후속 회고

대회 막바지에는 별도 로컬 확인에서 `tiiuae/Falcon-Perception-300M`이 일부 샘플에서 물병·박스 같은 객체명을 일반 VLM보다 더 직접적으로 출력하는 사례를 정성적으로 관찰했습니다. 하지만 GPU 예산과 일정상 object tag, crop 선택, pseudo-label 보정, routing feature, VQA 학습 입력으로 연결하는 단계까지는 진행하지 못했습니다.

따라서 이 저장소에서는 Falcon을 최종 제출 파이프라인이 아니라 후속 개선 아이디어로 정리합니다.

## 점수 해석

노트북명의 점수는 해당 실험 기록을 식별하는 실제 기록값입니다. 일부 제출은 로컬 저장명과 대회 업로드명이 다를 수 있어, README와 이 문서에서는 점수보다 실험 방법과 의사결정 흐름 중심으로 설명합니다.

- `score_0.94402.ipynb`는 선택적 Thinking 후처리 실험입니다.
- `ensemble_0.94363.ipynb`는 Qwen3.5-27B soft ensemble 실험입니다.
- `detection_tta_0.93102.ipynb`는 객체 인식 병목과 후반 검증 시간 부족을 보여주는 실패 포함 실험입니다.

## 원본 데이터와 weight

대회 원본 데이터와 학습된 모델 weight는 저장소에 포함하지 않습니다. 노트북은 실험 로직과 프로젝트 흐름을 설명하기 위한 대표 자료입니다.
