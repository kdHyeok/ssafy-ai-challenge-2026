# Experimental Notebooks

이 폴더에는 최종 핵심 흐름 외의 탐색/확장 실험을 보관합니다.

## `EDA.ipynb`

데이터 구조와 문제 특성을 파악하기 위한 노트북입니다.

확인한 축:
- train/test/dev 구조
- dev 응답 다수결과 agreement rate
- 질문 유형: count, material, color, text/brand 등
- pseudo-label 후보와 데이터 신뢰도
- image resize/padding/crop 등 입력 전처리 후보

EDA는 직접적인 성능 향상 claim이 아니라, 후속 32B 학습과 후처리 실험의 문제의식을 만든 근거로 해석합니다.

## `detection_tta_0.93102.ipynb`

대회 후반에 객체 인식 병목을 보완하려고 시도한 추론 보조 실험입니다.

확인된 흐름:
- Grounding DINO Base: 질문 키워드 기반 open-set detection 후보 생성
- Florence-2-large-ft: 이미지 전체 object detection 후보 생성
- 두 detector 결과를 score와 IoU 기준으로 병합
- best bbox 후보에 margin을 붙여 crop 생성
- detector는 언로드하고, 원본 이미지 + crop 이미지를 VQA 추론에 활용
- Qwen3-VL-32B LoRA와 Qwen3.5-27B의 원본/좌우반전 TTA 결과를 softmax 확률로 결합

주의할 점:
- detector 자체를 이 프로젝트에서 파인튜닝한 것은 아닙니다.
- detector 결과를 pseudo-label이나 fine-tuning label로 사용하지 않았고, VQA 모델 학습 데이터로 반영한 것도 아닙니다.
- 막바지에 추가 학습 시간이 부족해 추론에 덧붙인 방식이었고, 결과는 최고점권보다 낮은 후반 확장 실험으로 정리합니다.

## 한계와 후속 개선 방향

- `detection_tta_0.93102.ipynb`는 대회 막바지에 공개 detector를 inference-time crop 생성용으로 붙인 보조 전략이었습니다.
- 일부 샘플에 대한 별도 확인에서 `tiiuae/Falcon-Perception-300M`이 물병·박스 등 객체명을 일반 VLM보다 더 직접적으로 반환하는 사례를 정성적으로 봤습니다.
- Falcon 회고의 핵심은 최종 제출에 Falcon을 사용했다는 뜻이 아니라, 객체 인식 출력을 초반부터 object tag, crop 선택, pseudo-label 보정, question-type routing, VQA 학습 입력으로 연결했으면 좋았겠다는 점입니다.
- GPU 예산과 일정상 Falcon-Perception 결과를 VQA 학습 입력이나 추론 라우팅에 통합하는 단계까지는 진행하지 못했습니다.
