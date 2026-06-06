# Notebooks

이 디렉토리에는 VQA 대회에서 사용한 대표 노트북을 모았습니다. 상세한 기여 설명과 실패 기록은 `../docs/`에 따로 두었습니다.

## 파일 목록

| 파일 | 역할 | 내용 |
|---|---|---|
| `Baseline_0.70279.ipynb` | 초기 baseline | `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline. 데이터 로드부터 제출 CSV 생성까지 end-to-end 실행 |
| `train_0.9333.ipynb` | 32B 학습 | `Qwen/Qwen3-VL-32B-Instruct` + Unsloth/LoRA 학습. 파일명은 0.9333이지만 실제 제출 점수는 0.93141로 정리 |
| `ensemble_0.94363.ipynb` | soft ensemble | Qwen3-VL-32B와 Qwen3.5-27B를 결합해 a/b/c/d 확률을 가중 합산하는 실험 |
| `experimental/EDA.ipynb` | EDA | dev 응답 불일치, 질문 유형, pseudo-label, count/material 문제 분석 |
| `experimental/detection_tta_0.93102.ipynb` | detection/TTA 최종 제출 연결 코드 | 파일명은 0.93102지만, 제출 기록 재대조 결과 0.94402 연결 코드로 정리 |

## 관련 문서

- [모델 변경 실험 기록](../docs/model-change-experiment.md)
- [detection/TTA 최종 제출 기록](../docs/detection-tta-0.94402.md)
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

### 3. EDA와 데이터 진단

관련 파일: `experimental/EDA.ipynb`

- train/test/dev 구조 확인
- dev 응답 다수결과 agreement rate 분석
- 질문 유형 분류: count, material, color, text/brand
- 정답 분포와 선택지 편향 확인
- image resize/padding/crop이 count/material 문제에 어떤 영향을 주는지 실험 후보로 정리

이 단계는 모델 변경과 병행해서 진행했고, 이후 32B 학습·soft ensemble·detection/TTA 우선순위를 정하는 근거로 썼습니다.

### 4. 학습 파라미터 조정

관련 문서: `../docs/model-change-experiment.md`

- 초기 loss가 기대값과 비슷한지 확인
- 작은 샘플로 빠르게 overfit 되는지 확인
- `learning_rate`, `scheduler`, `label_smoothing`, `optimizer` 비교
- MiniCPM-V-4_5에서 특정 설정이 답변을 a로 고정시키는 문제 확인
- train loss와 eval loss를 같이 보고 다음 설정을 고름

### 5. Qwen3-VL-32B LoRA 학습

관련 파일: `train_0.9333.ipynb`

- 모델: `Qwen/Qwen3-VL-32B-Instruct`
- 프레임워크: Unsloth, TRL SFTTrainer, PEFT LoRA
- 실제 제출 점수: 0.93141
- 파일명 0.9333은 실제 제출 점수와 다름
- a/b/c/d별 softmax 확률 저장 구조를 포함

### 6. Soft ensemble

관련 파일: `ensemble_0.94363.ipynb`

- Model A: Qwen3-VL-32B
- Model B: Qwen3.5-27B
- 방식: a/b/c/d softmax 확률 가중 합산
- 결과: 0.94363

### 7. Detection / TTA 최종 제출

관련 파일: `experimental/detection_tta_0.93102.ipynb`

- Grounding DINO, Florence-2로 관심 객체 후보 crop 생성
- 원본 이미지와 crop 이미지를 같이 넣어 추론
- TTA와 듀얼 모델 확률 결합
- 파일명: 0.93102
- 제출 기록 재대조 결과 실제 연결 점수: 0.94402
- 자세한 기록은 `../docs/detection-tta-0.94402.md` 참고

### 8. 실패한 Thinking 실험

- 여러 제출이 갈리는 샘플만 다시 푸는 selective Thinking 아이디어
- 긴 로그를 텍스트 파일로 남긴 뒤 다시 파싱하는 구조
- 저장 로그와 최종 answer 정리가 불안정해 공개 레포에서는 노트북을 제거함
- 자세한 내용은 `../docs/selective-thinking-failed-experiment.md` 참고

## 점수 해석

노트북 파일명에 붙은 숫자와 실제 제출 점수가 다를 수 있습니다. 이 저장소에서는 `train_0.9333.ipynb → 0.93141`, `detection_tta_0.93102.ipynb → 0.94402`처럼 재검토된 값을 우선 적고, 실패한 selective Thinking 실험은 공개 레포에서 노트북을 제거한 뒤 문서만 남겼습니다.
