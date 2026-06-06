# `detection_tta_0.93102.ipynb` 최종 제출 기록

이 문서는 `../notebooks/experimental/detection_tta_0.93102.ipynb`를 따로 설명합니다. 파일명은 `0.93102`지만, 제출 기록을 다시 대조한 결과 실제 0.94402 연결 코드로 기록합니다.

## 한 줄 요약

- 파일명: `detection_tta_0.93102.ipynb`
- 이번 정리의 기준 역할: 0.94402 연결 코드
- 핵심 아이디어: detection crop + 원본 이미지 + TTA + 듀얼 VQA 확률 결합

## 실험 요약

| 항목 | 내용 |
|---|---|
| 관련 노트북 | `../notebooks/experimental/detection_tta_0.93102.ipynb` |
| detector | Grounding DINO, Florence-2 |
| VQA 축 | Qwen3-VL-32B LoRA, Qwen3.5-27B |
| 입력 방식 | 원본 이미지 + 관심 객체 crop 이미지 |
| 후처리 | 좌우반전 TTA, a/b/c/d softmax 확률 결합 |
| 이번 정리에서의 위치 | 최고점 0.94402 연결 코드 |

## 노트북이 하는 일

1. detector로 관심 객체 후보를 찾습니다.
2. 원본 이미지와 crop 이미지를 같이 준비합니다.
3. 두 VQA 축에서 답과 a/b/c/d 확률을 만듭니다.
4. 원본/좌우반전 TTA 결과를 합칩니다.
5. 최종적으로 확률을 결합해 제출 CSV를 만듭니다.

## 왜 다시 올려 적었는가

### 1. 파일명 숫자와 실제 연결 점수가 다릅니다

이 저장소에는 파일명 숫자와 실제 제출 점수가 어긋난 노트북이 있습니다. `train_0.9333.ipynb`도 실제 점수는 0.93141이고, 이 노트북도 파일명은 0.93102지만 제출 기록 대조 결과 0.94402 연결 코드로 적습니다.

### 2. 기존 공개 문서의 인과관계가 뒤집혀 있었습니다

이전 문서에서는 detection/TTA를 후반 확장 실험이나 실패 실험처럼 적은 부분이 있었습니다. 제출 기록을 다시 대조한 뒤 최고점 연결 축으로 다시 배치했습니다.

### 3. detector는 보조 모듈로 사용했습니다

이 노트북의 detector는 별도 추가학습한 전용 검출기가 아니라, 공개 detector를 추론 보조로 붙인 형태입니다. 즉 detector 출력을 학습 라벨로 다시 만들거나 초반 routing feature로 완전히 통합한 구조는 아닙니다.

## 이 실험에서 남는 의미

- count/material처럼 원본 한 장만으로 헷갈리는 샘플을 보완하려고 했습니다.
- 원본 이미지와 crop 이미지를 같이 넣어 객체 단서를 더 직접 보게 했습니다.
- 단일 모델 단답보다 확률 결합 구조를 더 적극적으로 사용했습니다.
- soft ensemble과 달리 detector/TTA를 같이 넣은 후반 파이프라인이라는 점에서 역할이 다릅니다.

## 함께 봐야 할 문서

- 전체 흐름: `../README.md`
- 노트북 요약: `../notebooks/README.md`
- 실패한 Thinking 실험: `selective-thinking-failed-experiment.md`
- 모델 변경과 학습 파라미터: `model-change-experiment.md`
