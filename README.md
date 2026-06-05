# SSAFY AI Challenge 2026 VQA

![Banner](Image/banner.png)

재활용품 이미지와 객관식 질문을 입력으로 받아 정답 선택지를 예측하는 VQA 대회 프로젝트입니다. Qwen3-VL-32B LoRA 학습을 중심으로 선택적 Thinking 재추론, 확률 기반 soft ensemble, detection/TTA 보조 실험을 비교했으며, 최고 0.94402점과 Private Leaderboard 12위를 기록했습니다.

## 주요 성과

- 최고 제출 점수: 0.94402
- 리더보드: Private Leaderboard 12위, 팀명 `C044_동준이와 아이들`
- 핵심 실험 축: Qwen3-VL-32B LoRA 학습, a/b/c/d 확률 저장, 선택적 Thinking 재추론
- 비교 실험: MiniCPM/Gemma/Qwen3-VL-8B 후보 검토, Qwen3.5-27B 계열 soft ensemble, detection/TTA 후반 확장 실험
- 공개 범위: split별 10행 CSV와 이미지 샘플, 대표 노트북, 결과 흐름 요약

![Private leaderboard snapshot](Image/private_12th.png)

## 문제 정의

- 입력: 재활용품 이미지 + 질문 + 보기 a/b/c/d
- 출력: 정답 선택지 한 글자
- 평가: 제출 CSV의 `id, answer` 기준 정확도
- 공개 데이터: 전체 원본 데이터 대신 `data/`에 train/dev/test split별 10행 CSV와 이미지 샘플 10장씩 포함

주요 난점은 이미지와 텍스트를 동시에 이해해야 한다는 점입니다. 모델은 재질, 개수, 형태, 라벨, 포장 상태처럼 시각적 세부 정보를 질문 문맥과 함께 판단해야 했습니다. 또한 dev 응답 불일치와 count/material 계열 질문은 단순 fine-tuning 점수만으로 해석하기 어려워, 데이터 신뢰도와 샘플별 불확실성을 함께 보는 방식으로 접근했습니다.

## 접근 방법 요약

![Pipeline](Image/pipeline.svg)

1. Qwen2.5-VL-3B-Instruct 기반 4bit LoRA baseline으로 데이터 로드, 프롬프트 구성, 학습, 제출 CSV 생성까지 end-to-end 흐름을 확보했습니다.
2. MiniCPM, Gemma, Qwen3-VL-8B 계열을 후보로 검토하며 모델 크기, 입력 포맷, 실행 가능성, 제출 점수 흐름을 비교했습니다. google/gemma-4-E4B-it는 0.90815, Qwen3-VL-8B는 0.91643으로 기록되어 Qwen3 계열 확장 판단의 근거가 되었습니다.
3. EDA를 통해 질문 유형, 정답 분포, dev 응답 일치도, count/material 계열 문제를 확인했습니다.
4. Qwen3-VL-32B LoRA 학습으로 확장하고, answer뿐 아니라 a/b/c/d 확률을 저장해 후속 재추론과 ensemble 비교가 가능하도록 했습니다.
5. 여러 제출에서 답이 갈리거나 confidence가 낮은 샘플을 대상으로 선택적 Thinking 재추론을 적용했습니다.
6. 후반에는 Qwen3.5-27B 계열 soft ensemble과 공개 detector 기반 detection/TTA 보조 실험을 비교했습니다.

## 프로젝트 구조

```text
ssafy-ai-challenge-2026-vqa/
├── README.md
├── Image/
│   ├── banner.png
│   ├── private_12th.png
│   ├── pipeline.svg
│   └── score.svg
├── data/
│   ├── README.md
│   ├── train.csv / dev.csv / test.csv  # split별 10행 샘플
│   ├── sample_submission_format.csv    # test 10행 제출 형식 예시
│   ├── train/  # train 이미지 샘플 10장
│   ├── dev/    # dev 이미지 샘플 10장
│   └── test/   # test 이미지 샘플 10장
└── notebooks/
    ├── README.md
    ├── Baseline_0.70279.ipynb
    ├── train_0.9333.ipynb
    ├── score_0.94402.ipynb
    ├── ensemble_0.94363.ipynb
    └── experimental/
        ├── README.md
        ├── EDA.ipynb
        └── detection_tta_0.93102.ipynb
```

## 실험 흐름

### 1. 초기 baseline과 모델 후보군 탐색

처음에는 Qwen2.5-VL-3B-Instruct 기반 4bit LoRA baseline으로 데이터 로드, 이미지-질문-보기 프롬프트 구성, 학습 루프, 제출 CSV 생성까지 확인했습니다. 이 baseline은 0.70279 수준의 초기 기준선으로, 이후 후보 모델과 실험 구조를 비교하기 위한 출발점이 되었습니다.

이후 MiniCPM-V-4_5, google/gemma-4-E4B-it, Qwen3-VL-8B-Instruct 등으로 후보군을 넓혀 비교했습니다. MiniCPM 계열은 제출 기록상 0.85~0.86대 후보군으로 관찰되었고, google/gemma-4-E4B-it는 0.90815, Qwen3-VL-8B는 0.91643을 기록했습니다. 이 흐름을 바탕으로 Qwen3 계열의 입력 포맷과 실행 가능성을 먼저 확인한 뒤, 더 큰 32B 모델로 확장했습니다.

### 2. EDA와 데이터 진단

EDA에서는 train/test/dev 파일을 단순 통계로만 본 것이 아니라, 질문 유형, 정답 분포, dev 응답 다수결, agreement rate, count/material/color/text 계열 문제를 나누어 분석했습니다.

이 분석은 직접적인 성능 향상 claim이 아니라, 이후 32B 학습, pseudo-label 후보 검토, confidence 기반 후처리, detection crop 실험의 우선순위를 정하는 데 활용했습니다.

### 3. Qwen3-VL-32B LoRA 학습

3B baseline과 초기 후보군 검토 이후, 더 큰 VLM 계열을 활용할 필요가 있다고 보고 Qwen3-VL-32B LoRA 학습으로 확장했습니다. 32B 단일 모델을 별도 순수 baseline으로 제출해 비교하기보다는, 모델 채택 직후 전처리, pseudo-label 후보 활용, augmentation, 확률 저장 구조까지 함께 적용해 실험을 진행했습니다. 대표 노트북 `train_0.9333.ipynb`는 Unsloth 기반 32B 학습, LoRA 설정 조정, epoch/lr/batch 관련 제한적 튜닝을 포함합니다.

이 단계부터 단순히 answer만 저장하는 방식에서 벗어나 a/b/c/d 확률을 함께 저장했습니다. 이 구조는 이후 선택적 Thinking 재추론과 soft ensemble의 입력 자료가 되었습니다.

### 4. 선택적 Thinking 재추론

`score_0.94402.ipynb`는 여러 제출 결과가 일치하는 쉬운 샘플과 답이 갈리는 저합의 샘플을 나누고, 저합의 샘플에 Qwen3-VL-32B-Thinking 계열을 적용하는 후처리 실험입니다.

이 계열은 최고점 0.94402 제출과 강하게 연결됩니다. 다만 Thinking만의 독립 기여량을 별도 ablation으로 분리하지 않았고, 대회 후반에는 Thinking 전체 적용이 오히려 떨어지는 경우도 있어, 이 저장소에서는 저합의 샘플 대상 선택적 보정 실험으로 설명합니다.

### 5. Soft ensemble과 confidence routing

`ensemble_0.94363.ipynb` 계열은 Qwen3-VL-32B를 먼저 추론하고, confidence가 낮은 샘플에 Qwen3.5-27B 계열 보조 모델을 선택적으로 적용한 뒤 a/b/c/d 확률을 가중 결합하는 듀얼 모델 soft ensemble입니다.

soft ensemble 계열 제출은 0.941~0.943대 점수로 기록되었지만, 단일 32B 계열 대비 일관된 우위를 보였다고 단정하지 않습니다. 이 실험은 확률 저장 구조와 confidence routing을 활용한 후반 비교 실험으로 정리합니다.

### 6. Detection/TTA와 Falcon 회고

후반에는 count, 작은 물체, 객체 인식 문제를 보완하기 위해 detection crop과 TTA를 붙이는 실험을 진행했습니다. `detection_tta_0.93102.ipynb`는 Grounding DINO와 Florence-2 공개 detector를 별도 학습 없이 사용해 관심 객체 후보 crop을 만들고, 원본 이미지와 crop 이미지를 VQA 추론 입력에 함께 활용한 실험입니다.

이 계열은 0.926~0.931대에 머물러 최고점 핵심 전략으로 이어지지는 않았습니다. 다만 VLM 단독 추론에서 남는 객체 인식 병목을 확인한 후반 확장 실험으로 의미가 있었습니다.

Falcon-Perception은 최종 제출 파이프라인에 포함되지 않았습니다. 일부 샘플 정성 확인에서 객체명 반환 가능성을 봤고, 이를 object tag, crop routing, pseudo-label 보정, VQA 학습 입력으로 초반부터 연결했으면 좋았겠다는 후속 개선 아이디어로 정리했습니다.

## 대표 결과

![Score](Image/score.svg)

| 구분 | 대표 결과 | 해석 |
|---|---:|---|
| 최고 제출군 | 0.94402 | 32B 중심 추론과 선택적 Thinking 재추론 계열이 최고점권 결과로 연결됨. 단일 후처리만의 독립 기여로 단정하지 않음 |
| 리더보드 캡처 | Private 12위 / 0.94402 | `Image/private_12th.png`에 보관된 공개 가능한 결과 근거 |
| Thinking selective replace | 0.94402 | 저합의 샘플 대상 선택적 재추론 계열. 전체 Thinking 적용의 일반 성공으로 해석하지 않음 |
| 듀얼/soft ensemble | 0.941~0.943 | 여러 제출에서 고득점 후보군으로 기록됐지만 최고점은 넘지 못함 |
| Qwen3-VL-32B 학습/추론 기반 | 후속 고득점 제출군의 기반 | 32B LoRA 학습과 확률 기반 추론 저장 구조가 후속 전략의 입력 자료 |
| Detection/TTA 계열 | 0.926~0.931 | 후반 추론 보조 확장 실험. 복잡도 대비 최고점 기여는 제한적 |
| 초기 baseline/후보군 검토 | 0.70279 → 0.85~0.91643 | Qwen2.5 3B baseline에서 출발해 MiniCPM 0.85~0.86대, google/gemma-4-E4B-it 0.90815, Qwen3-VL-8B 0.91643을 검토하고 Qwen3 계열 32B 확장으로 이어진 단계 |

## 내가 기여한 부분

팀 프로젝트에서 저는 데이터 분석, 초기 모델 후보 검토, 32B 계열 실험 확장, 확률 기반 추론 저장 구조, 선택적 재추론 방향 정리에 기여했습니다. 최종 점수는 팀의 반복 실험 결과로 보고, 아래에는 제가 직접 작성하거나 제안한 작업을 중심으로 정리했습니다.

| 영역 | 기여 내용 | 산출물 |
|---|---|---|
| EDA 및 데이터 분석 | dev 응답 불일치, 질문 유형, count/material 계열 문제를 분석하고 후속 실험 우선순위를 정리 | `notebooks/experimental/EDA.ipynb` |
| 초기 모델 검토 | Qwen2.5-VL-3B baseline 이후 MiniCPM, Gemma, Qwen3-VL-8B 후보를 검토하고 32B 확장 방향을 제안 | `notebooks/Baseline_0.70279.ipynb`, `notebooks/train_0.9333.ipynb` |
| 32B 학습 실험 | Qwen3-VL-32B LoRA 학습 구성과 pseudo-label, augmentation, 확률 저장 흐름을 정리 | `notebooks/train_0.9333.ipynb` |
| 추론 결과 저장 구조 | answer만 저장하지 않고 a/b/c/d 확률을 함께 저장하는 포맷을 제안해 후속 ensemble과 재추론 비교가 가능하도록 함 | `notebooks/score_0.94402.ipynb`, `notebooks/ensemble_0.94363.ipynb` |
| 선택적 재추론 | 제출 간 answer disagreement와 낮은 confidence 샘플을 대상으로 Thinking 재추론을 적용하는 방향을 제안 | `notebooks/score_0.94402.ipynb` |
| 실패 실험 정리 | detection/TTA 등 최고점으로 이어지지 않은 후반 실험도 점수와 한계를 분리해 정리 | `notebooks/experimental/detection_tta_0.93102.ipynb` |

## 주요 트러블슈팅과 배운 점

### GPU 자원 제약

32B급 VLM은 성능 가능성이 있었지만, 학습 시간이 길고 OOM이 쉽게 발생했습니다. 모델을 무리하게 크게만 쓰기보다 4bit 양자화와 LoRA를 기준으로 실험 가능성을 먼저 확보했고, batch size, gradient accumulation, 학습 스케줄을 조정하며 반복 가능한 실험 단위를 만들었습니다.

### 데이터 품질과 dev 활용 문제

dev 데이터는 정답이 항상 깔끔하게 일치하지 않았고, 질문 유형 분포도 test 입력과 다를 수 있었습니다. dev를 그대로 버리거나 그대로 합치기보다, 응답 일치도와 질문 유형을 기준으로 신뢰할 수 있는 샘플을 나누고 test 입력의 질문 유형·이미지 특성 분포를 참고해 활용하는 방향을 검토했습니다.

### 후처리 구조와 선택적 재추론

제출 파일에 최종 answer만 남기면, 모델들이 어디서 흔들리는지 비교하거나 앙상블을 만들기 어려웠습니다. 그래서 a/b/c/d 확률을 함께 저장하는 포맷을 제안했고, 이를 바탕으로 confidence가 낮거나 제출 간 답이 갈리는 샘플을 따로 보는 흐름을 만들었습니다.

### VLM의 객체 인식 한계

대회 막바지에는 일반 VLM이 재활용품 객체를 세밀하게 인식하지 못하는 문제가 계속 남아 있었습니다. detection/TTA는 공개 detector로 crop 후보를 만들어 추론 입력에 보조로 붙인 방식이었고, 객체 인식 결과를 학습 데이터나 모델 앞단에 통합하지는 못했습니다. 다음에는 객체 인식 결과를 초반부터 학습 입력이나 샘플 분기 기준으로 활용해야 한다는 교훈을 얻었습니다.

## 성장한 점

- VLM fine-tuning 프로젝트를 baseline, EDA, 학습, 추론, 후처리, 제출, 회고까지 end-to-end로 경험했습니다.
- 32B급 모델을 제한된 GPU 환경에서 실행하기 위해 LoRA/QLoRA, 4bit quantization, Unsloth 최적화를 적용했습니다.
- 데이터 품질과 dev 활용 방식이 모델 성능만큼 중요하다는 것을 체감했습니다.
- 고득점 결과만 보여주는 것이 아니라, 효과가 제한적인 실험과 한계를 분리해 설명하는 포트폴리오 서사의 중요성을 배웠습니다.
- 팀 대회에서 실험 로그, 제출명, 점수 흐름을 관리하는 것이 후반 의사결정과 회고 품질에 직접 영향을 준다는 것을 경험했습니다.

## 문서 안내

- [대표 노트북 설명](notebooks/README.md)
- [데이터 샘플 구조](data/README.md)

## 재현 범위와 공개 범위

학습된 모델 weight와 checkpoint는 포함하지 않습니다. 공개 노트북은 실험 흐름과 핵심 로직을 보여주기 위한 대표 파일입니다. `data/`에는 CSV 구조와 이미지 경로를 설명하기 위한 split별 10행 샘플과 이미지 샘플만 두었으며, 전체 데이터가 포함된 완전 재현용 데이터셋은 아닙니다. 따라서 공개 레포만으로 exact score 재현을 보장하지는 않고, 대표 모델명·핵심 학습 코드·추론 로직·점수 흐름을 검토할 수 있게 정리했습니다.

노트북은 Colab과 GPU 환경에서 수행된 대표 실험 기록이며, 각 노트북의 설치 셀과 실행 환경에 의존합니다. 단일 패키지 목록만으로 전체 대회 환경을 재현할 수 있다고 오해되지 않도록, 공개 루트에는 별도 환경 파일을 두지 않았습니다.

## License

이 저장소는 SSAFY AI Challenge 수행 내용을 포트폴리오/회고 목적으로 정리한 공개용 저장소입니다. 코드와 문서의 재사용 범위는 별도 `LICENSE` 파일을 추가해 명확히 정리하는 것을 권장합니다.
