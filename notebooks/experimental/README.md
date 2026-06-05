# Experimental Notebooks

이 폴더에는 최종 핵심 흐름 외의 탐색/확장 실험을 보관합니다.

## `EDA.ipynb`

데이터 구조와 문제 특성을 파악하기 위한 노트북입니다.

확인한 축:
- train/dev/test 구조
- dev 응답 다수결과 agreement rate
- 질문 유형: count, material, color, text/brand
- 정답 분포와 선택지 편향
- image resize, padding, crop이 count/material 문제와 연결될 가능성

EDA는 직접적인 성능 향상 claim이 아니라, 32B 학습, confidence routing, Thinking 재추론, detection crop 실험의 문제의식으로 정리합니다.

## `detection_tta_0.93102.ipynb`

최종 핵심 전략이 아니라 후반 확장 실험입니다.

실험 내용:
- Grounding DINO와 Florence-2 공개 detector로 관심 객체 후보 crop 생성
- detector는 별도 추가학습 없이 추론 보조 모듈로 사용
- 원본 이미지와 crop 이미지를 함께 VQA 입력으로 사용
- Qwen3-VL-32B LoRA와 Qwen3.5-27B의 원본/좌우반전 TTA 결과를 softmax 확률로 결합
- 결과: 0.93102

해석:
- VLM 단독 추론이 재활용품 객체를 세밀하게 인식하지 못하는 문제를 보완하려는 시도였습니다.
- detector 결과를 pseudo-label이나 fine-tuning label로 사용하지 않았고, 추론 시 crop/TTA 보조 입력으로만 사용했습니다.
- 복잡도 대비 최고점 개선으로 이어지지는 않았습니다.

## Falcon-Perception 회고

`tiiuae/Falcon-Perception-300M`은 일부 샘플에서 물병·박스 같은 객체명을 더 직접적으로 반환하는 사례를 확인했습니다. 하지만 대회 후반에 확인한 수준이었고, object tag, crop 선택, pseudo-label 보정, routing feature, VQA 학습 입력으로 연결하지 못했습니다.

다음에 비슷한 VQA 대회를 한다면 객체 인식 결과를 후반 보조 실험으로 붙이는 것보다, 초반부터 학습 입력과 샘플 분기 기준에 통합하는 방향을 먼저 검토할 것입니다.
