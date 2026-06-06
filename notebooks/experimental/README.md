# Experimental Notebooks

이 폴더에는 탐색 실험과 후반 파이프라인 노트북을 보관합니다.

## `EDA.ipynb`

데이터 구조와 문제 특성을 파악하기 위한 노트북입니다.

확인한 내용:
- train/dev/test 구조
- dev 응답 다수결과 agreement rate
- 질문 유형: count, material, color, text/brand
- 정답 분포와 선택지 편향
- image resize, padding, crop이 count/material 문제에 어떤 영향을 주는지

EDA는 직접적인 점수 설명보다 후속 실험 우선순위를 잡는 자료로 보는 편이 맞습니다.

## `detection_tta_0.93102.ipynb`

이 노트북은 detection crop, TTA, 듀얼 VQA 결합을 한 후반 파이프라인입니다.

실험 내용:
- Grounding DINO와 Florence-2 공개 detector로 관심 객체 후보 crop 생성
- detector는 별도 추가학습 없이 추론 보조 모듈로 사용
- 원본 이미지와 crop 이미지를 함께 VQA 입력으로 사용
- Qwen3-VL-32B LoRA와 Qwen3.5-27B의 원본/좌우반전 TTA 결과를 softmax 확률로 결합
- 파일명: 0.93102
- 제출 기록 재대조 결과 실제 연결 점수: 0.94402

설명:
- 이전 문서에서는 이 노트북을 후반 실패 실험처럼 적은 부분이 있었지만, 제출 기록을 다시 대조한 결과 최고점 연결 코드로 정리했습니다.
- 따라서 이 폴더에서는 제거된 selective Thinking 실패 실험보다 이 노트북을 더 중요한 제출 코드로 봅니다.
- 자세한 기록은 [`../../docs/detection-tta-0.94402.md`](../../docs/detection-tta-0.94402.md)에 따로 정리했습니다.

## selective Thinking 실패 실험과의 구분

실패한 selective Thinking 실험 노트북은 공개 레포에서 제거했습니다.
파일명만 보고 최고점 코드로 읽히지 않게 [`../../docs/selective-thinking-failed-experiment.md`](../../docs/selective-thinking-failed-experiment.md)에 따로 기록했습니다.

## Falcon-Perception 회고

`tiiuae/Falcon-Perception-300M`은 일부 샘플에서 물병·박스 같은 객체명을 더 직접 반환하는 사례가 있었습니다. 하지만 object tag, crop 선택, pseudo-label 보정, routing feature, VQA 학습 입력으로 연결하지 못했습니다.

다음에 비슷한 VQA 대회를 한다면 객체 인식 결과를 후반에 보조로 붙이기보다, 초반부터 학습 입력과 샘플 분기 기준에 넣는 방향을 먼저 검토할 것입니다.
