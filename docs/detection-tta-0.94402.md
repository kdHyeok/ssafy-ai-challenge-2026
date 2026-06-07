# `detection_tta_0.94402.ipynb` 최종 제출 기록

이 문서는 `../notebooks/detection_tta_0.94402.ipynb`를 따로 설명합니다. detection crop, TTA, 듀얼 VQA 결합을 사용한 최종 제출 파이프라인을 기록합니다.

## 한 줄 요약

- 관련 노트북: `detection_tta_0.94402.ipynb`
- 역할: 0.94402 최종 제출 파이프라인
- 핵심 아이디어: detection crop + 원본 이미지 + TTA + 듀얼 VQA 확률 결합

## 실험 요약

| 항목 | 내용 |
|---|---|
| 관련 노트북 | `../notebooks/detection_tta_0.94402.ipynb` |
| detector | Grounding DINO, Florence-2 |
| VQA 축 | Qwen3-VL-32B LoRA, Qwen3.5-27B |
| 입력 방식 | 원본 이미지 + 관심 객체 crop 이미지 |
| 후처리 | 좌우반전 TTA, a/b/c/d softmax 확률 결합 |
| 위치 | 0.94402 최종 제출 파이프라인 |

## 노트북이 하는 일

1. detector로 관심 객체 후보를 찾습니다.
2. 원본 이미지와 crop 이미지를 같이 준비합니다.
3. 두 VQA 축에서 답과 a/b/c/d 확률을 만듭니다.
4. 원본/좌우반전 TTA 결과를 합칩니다.
5. 최종적으로 확률을 결합해 제출 CSV를 만듭니다.

## 이 파이프라인의 위치

### 1. 최종 제출에 사용한 흐름입니다

이 저장소에서는 실제 역할이 드러나도록 최종 제출 노트북을 `detection_tta_0.94402.ipynb`로 정리했습니다.

### 2. detector는 보조 모듈로 사용했습니다

이 노트북의 detector는 별도 추가학습한 전용 검출기가 아니라, 공개 detector를 추론 보조로 붙인 형태입니다. 즉 detector 출력을 학습 라벨로 다시 만들거나 초반 routing feature로 완전히 통합한 구조는 아닙니다.

## 이 실험에서 남는 의미

- count/material처럼 원본 한 장만으로 헷갈리는 샘플을 보완하려고 했습니다.
- 원본 이미지와 crop 이미지를 같이 넣어 객체 단서를 더 직접 보게 했습니다.
- 단일 모델 단답보다 확률 결합 구조를 더 적극적으로 사용했습니다.
- soft ensemble과 달리 detector/TTA를 같이 넣은 최종 제출 파이프라인이라는 점에서 역할이 다릅니다.

## 함께 봐야 할 문서

- 전체 흐름: `../README.md`
- 노트북 요약: `../notebooks/README.md`
- 실패한 Thinking 실험: `selective-thinking-failed-experiment.md`
- 모델 변경과 학습 파라미터: `model-change-experiment.md`
