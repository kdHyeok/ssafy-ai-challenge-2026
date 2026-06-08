# Notebooks

이 디렉토리에는 VQA 대회에서 사용한 대표 노트북을 모았습니다. 상세한 배경과 회고는 `../docs/`에 따로 두었습니다.

## 파일 목록

| 파일 | 역할 | 내용 |
|---|---|---|
| `Baseline_0.70279.ipynb` | 초기 baseline | `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline. 데이터 로드부터 제출 CSV 생성까지 end-to-end 실행 |
| `train_0.9333.ipynb` | 32B 학습 | `Qwen/Qwen3-VL-32B-Instruct` + Unsloth/LoRA 학습. 결과 0.93141 |
| `ensemble_0.94363.ipynb` | soft ensemble | Qwen3-VL-32B와 Qwen3.5-27B를 결합해 a/b/c/d 확률을 가중 합산하는 실험 |
| `final_score_0.94402.ipynb` | 최종 제출 파이프라인 | Grounding DINO, Florence-2, Qwen3-VL-32B, Qwen3.5-27B를 묶어 최종 제출에 사용한 추론 파이프라인 |
| `experimental/EDA.ipynb` | EDA | 데이터 구조, 질문 유형, 응답 불일치, 개수/재질 문제를 먼저 확인한 탐색 노트북 |

## 관련 문서

- [대회 개요](../docs/Overview.md)
- [Dataset Description](../docs/Dataset%20Description.md)
- [모델 변경 실험 기록](../docs/model-change-experiment.md)
- [detection/TTA 최종 제출 기록](../docs/detection-tta-0.94402.md)
- [객체 인식 회고](../docs/detection-retrospective.md)
- [selective Thinking 실패 실험 기록](../docs/selective-thinking-failed-experiment.md)

## 실험 순서

### 1. Baseline 확보

관련 파일: `Baseline_0.70279.ipynb`

- 모델: `Qwen/Qwen2.5-VL-3B-Instruct`
- 설정: 4bit LoRA, `IMAGE_SIZE=384`, LoRA `r=8`, `lora_alpha=16`, train 200개 샘플
- 결과: 0.70279

### 2. 모델 변경 실험

관련 문서: `../docs/model-change-experiment.md`

| 모델 | 실제 점수 | 판단 |
|---|---:|---|
| `openbmb/MiniCPM-V-4_5` | 0.78341 | baseline보다 높았지만 Qwen3-VL-8B보다 낮음 |
| `google/gemma-4-E4B-it` | 0.82815 | 최신 모델이었지만 Qwen3-VL-8B보다 낮음 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | 가장 높아 32B 확장의 기준으로 채택 |

### 3. EDA와 데이터 확인

관련 파일: `experimental/EDA.ipynb`

- train/dev/test가 어떤 구조인지 먼저 확인
- dev에서 답변이 얼마나 자주 갈리는지 확인
- 질문을 개수, 재질, 색상, 문자처럼 이해하기 쉬운 범주로 나눠 봄
- 정답 분포와 보기 쏠림이 있는지 확인
- 이미지 resize, padding, crop이 개수·재질 문제에 어떤 영향을 줄지 후속 실험 후보로 확인

이 단계는 모델 변경과 병행했고, 이후 32B 학습, soft ensemble, detection/TTA 실험 순서를 정하는 기준으로 사용했습니다.

### 4. 학습 파라미터 조정

관련 문서: `../docs/model-change-experiment.md`

- 초기 loss가 기대 범위에 있는지 확인
- 작은 샘플에서 빠르게 overfit 되는지 확인
- `learning_rate`, `scheduler`, `label_smoothing`, `optimizer` 비교
- MiniCPM-V-4_5에서 답이 한쪽으로 쏠리는 문제 확인
- train loss와 eval loss를 같이 보고 다음 설정을 고름

### 5. Qwen3-VL-32B LoRA 학습

관련 파일: `train_0.9333.ipynb`

- 모델: `Qwen/Qwen3-VL-32B-Instruct`
- 프레임워크: Unsloth, TRL SFTTrainer, PEFT LoRA
- 결과: 0.93141
- a/b/c/d별 softmax 확률 저장 구조 포함

### 6. Soft ensemble

관련 파일: `ensemble_0.94363.ipynb`

- Model A: Qwen3-VL-32B
- Model B: Qwen3.5-27B
- 방식: a/b/c/d softmax 확률 가중 합산
- 결과: 0.94363

### 7. Detection / TTA 최종 제출

관련 파일: `final_score_0.94402.ipynb`

- Grounding DINO와 Florence-2로 관심 객체 후보를 따로 찾음
- 원본 이미지와 crop 이미지를 함께 넣어 추론
- TTA와 듀얼 모델 확률 결합 사용
- 자세한 코드 흐름은 `../docs/detection-tta-0.94402.md` 참고
- 왜 객체 인식을 따로 보게 되었는지는 `../docs/detection-retrospective.md` 참고

### 8. 실패한 Thinking 실험

- 여러 제출이 갈리는 샘플만 다시 푸는 selective Thinking 아이디어
- 긴 로그를 텍스트 파일로 남긴 뒤 다시 파싱하는 구조
- 저장 로그와 최종 answer 생성이 불안정해 최종 제출 흐름에는 포함하지 않음
- 자세한 내용은 `../docs/selective-thinking-failed-experiment.md` 참고
