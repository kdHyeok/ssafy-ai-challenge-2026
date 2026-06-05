# Notebooks

이 디렉토리는 VQA 대회 실험 흐름을 설명하는 대표 노트북을 포함합니다. baseline, EDA, 32B LoRA 학습, 선택적 재추론, soft ensemble, detection/TTA 확장 실험 순서로 구성했습니다.

## 대표 노트북

| 파일 | 역할 | 핵심 내용 |
|---|---|---|
| `Baseline_0.70279.ipynb` | 초기 baseline | Qwen2.5-VL-3B-Instruct 기반 4bit LoRA baseline. 데이터 로드부터 제출 CSV 생성까지 end-to-end 흐름 확인 |
| `train_0.9333.ipynb` | 32B 학습 | Qwen3-VL-32B-Instruct + Unsloth/LoRA 학습. 8B 후보 검토 이후 32B로 확장하고, pseudo-label, augmentation, a/b/c/d 확률 저장을 함께 적용한 실험 계열. 파일명의 0.9333은 해당 32B 학습 계열의 기록값 |
| `score_0.94402.ipynb` | 선택적 Thinking 재추론 | 여러 제출 사이에서 답이 갈리는 저합의 샘플을 Qwen3-VL-32B-Thinking 모델로 재추론하는 후처리 실험 |
| `ensemble_0.94363.ipynb` | 듀얼 모델 soft ensemble | Qwen3-VL-32B 전체 추론 후 저신뢰 샘플에 Qwen3.5-27B를 선택적으로 적용하고, a/b/c/d 확률을 가중 결합하는 고득점 후보군 실험 |
| `experimental/EDA.ipynb` | EDA | dev 응답 불일치, 질문 유형, pseudo-label, count/material 문제 등 분석 |
| `experimental/detection_tta_0.93102.ipynb` | 후반 확장 실험 | 공개 detector 기반 crop 생성, 원본+crop 입력, TTA, VQA 확률 결합 실험 |

## 실험 흐름

### 1. Baseline 확보

대표 파일: `Baseline_0.70279.ipynb`

- Qwen2.5-VL-3B-Instruct 4bit LoRA baseline
- Colab GPU에서 데이터 로드, 이미지 전처리, 프롬프트 구성, fine-tuning, submission 생성까지 확인
- 0.70279 수준의 초기 기준선을 확보해 이후 모델 크기와 학습 전략을 비교할 출발점으로 사용

### 2. 모델 후보군 검토

- Qwen2.5-VL-3B baseline 이후 MiniCPM-V-4_5, google/gemma-4-E4B-it, Qwen3-VL-8B-Instruct 등을 초기 후보로 검토
- 이 단계의 목적은 모든 조합을 정량 ablation하는 것이 아니라, 입력 포맷·메모리 요구량·추론/학습 가능성을 확인하고 32B 확장 여부를 판단하는 것이었음
- MiniCPM 계열은 0.85~0.86대 후보군으로 baseline 대비 강한 비교 축으로 활용
- google/gemma-4-E4B-it는 0.90815, Qwen3-VL-8B는 0.91643을 기록해 Qwen3 계열 확장의 중간 근거로 활용
- loss function, scheduler, label smoothing, focal loss 등은 핵심 성과 claim이 아니라 학습 안정화를 위해 검토한 후보로만 정리

### 3. EDA와 데이터 진단

대표 파일: `experimental/EDA.ipynb`

- train/test/dev 구조 확인
- dev 응답 다수결과 agreement rate 분석
- 질문 유형 분류: count, material, color, text/brand 등
- 정답 분포와 선택지 편향 확인
- image resize/padding/crop이 count/material 문제와 어떤 관련이 있을지 후속 실험 후보로 정리

EDA는 직접적인 성능 향상 claim이 아니라, 이후 32B 학습, confidence routing, Thinking 재추론, detection crop 실험의 문제의식으로 해석합니다.

### 4. Qwen3-VL-32B LoRA 학습

32B 실험은 갑작스러운 모델 교체가 아니라, 초기 후보군 검토에서 Qwen3-VL-8B-Instruct가 0.91643을 기록하며 같은 계열의 입력 포맷과 실행 흐름을 먼저 점검한 뒤 더 큰 모델을 검토한 흐름의 연장선입니다. 32B 단일 모델을 별도 순수 baseline으로 제출해 비교하기보다는, 모델 채택 후 전처리와 확률 저장 구조를 함께 적용해 실험을 진행했습니다.

대표 파일: `train_0.9333.ipynb`

- 모델: `Qwen/Qwen3-VL-32B-Instruct`
- 프레임워크: Unsloth, TRL SFTTrainer, PEFT LoRA
- 학습 전략: train 전체와 dev pseudo-label 후보 활용, augmentation, sequence length 점검
- 조정한 주요 축: epoch, learning rate, LoRA rank/alpha, pseudo-label agreement 기준
- batch size와 gradient accumulation은 성능 탐색뿐 아니라 4bit/gradient checkpointing 환경에서 32B 모델을 올리기 위한 메모리 제약 대응 성격이 큼
- a/b/c/d별 softmax 확률 저장으로 선택적 재추론과 soft ensemble의 비교 기준 마련

### 5. 선택적 Thinking 재추론

대표 파일: `score_0.94402.ipynb`

- 모델: `unsloth/Qwen3-VL-32B-Thinking-unsloth-bnb-4bit`
- 여러 제출이 같은 답을 고른 쉬운 샘플과 답이 갈리는 저합의 샘플 분리
- 여러 제출 간 answer disagreement 또는 낮은 confidence가 있는 저합의 샘플에만 Thinking 모델을 적용해 기존 제출 일부를 교체
- 대표 산출물: `main-sub-think-2.csv`

Thinking은 모든 문제에 일괄 적용한 전략이 아니라, 불확실한 샘플을 대상으로 한 후처리 전략입니다. 최고점 0.94402 계열과 강하게 연결되지만, Thinking만의 독립 기여량은 별도 ablation으로 분리하지 않았습니다.

### 6. 듀얼 모델 soft ensemble

대표 파일: `ensemble_0.94363.ipynb`

- Model A: Qwen3-VL-32B 계열
- Model B: Qwen3.5-27B 계열
- 방식: a/b/c/d softmax 확률을 가중 합산
- 일부 노트북은 confidence가 낮은 샘플에만 보조 모델을 호출

이 전략은 hard voting이 아니라 확률 기반 soft ensemble입니다. 여러 제출에서 0.941~0.943대 후보군으로 기록됐지만, 단일 32B 계열 대비 일관된 우위를 보였다고 단정하지 않습니다.

### 7. Detection/TTA 확장

대표 파일: `experimental/detection_tta_0.93102.ipynb`

- Grounding DINO, Florence-2 공개 detector로 test 이미지의 관심 객체 후보 crop 생성
- detector는 별도 추가학습 없이 추론 보조 모듈로 사용
- 원본 이미지와 crop 이미지를 VQA 추론 입력에 함께 활용
- Qwen3-VL-32B LoRA와 Qwen3.5-27B의 원본/좌우반전 TTA 확률 결합

이 계열은 VLM이 재활용품 객체를 세밀하게 인식하지 못하는 문제를 보완하려는 후반 확장 실험입니다. detector 결과를 pseudo-label이나 fine-tuning label로 사용하지 않았고, 추론 시 crop/TTA 보조 입력으로만 사용했습니다. 점수는 0.926~0.931대에 머물러 복잡도 대비 점수 이득은 제한적이었습니다.

### 8. Falcon-Perception 후속 회고

대회 막바지에는 별도 로컬 확인에서 `tiiuae/Falcon-Perception-300M`이 일부 샘플에서 물병·박스 같은 객체명을 일반 VLM보다 더 직접적으로 출력하는 사례를 정성적으로 관찰했습니다. 하지만 GPU 예산과 일정상 object tag, crop 선택, pseudo-label 보정, routing feature, VQA 학습 입력으로 연결하는 단계까지는 진행하지 못했습니다.

따라서 이 저장소에서는 Falcon을 최종 제출 파이프라인이 아니라 “detection을 후반 추론 보조로만 붙인 한계에서 출발한 후속 개선 아이디어”로 정리합니다.

## 점수와 공개 범위

노트북명의 점수는 해당 실험 기록을 식별하는 실제 기록값입니다. 일부 제출은 로컬 저장명과 대회 업로드명이 다를 수 있어, README와 이 문서에서는 점수보다 실험 계열과 방법 중심으로 설명합니다.

- `score_0.94402.ipynb`는 선택적 Thinking 후처리 실험입니다.
- 32B 중심 제출군은 최고점권 기반 계열로 설명하되, 단일 로컬 CSV와 exact 동일하다고 단정하지 않습니다.
- detection/TTA는 최고점 핵심 전략이 아니라, 객체 인식 병목과 후반 검증 시간 부족을 보여주는 실패 포함 실험입니다.

## 원본 데이터와 weight

대회 원본 데이터와 학습된 모델 weight는 저장소에 포함하지 않습니다. 노트북은 실험 로직과 프로젝트 흐름을 설명하기 위한 대표 자료입니다.
