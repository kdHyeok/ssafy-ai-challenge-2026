# SSAFY AI Challenge 2026 VQA

![Banner](Image/banner.png)

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [주요 정리](#주요-정리)
- [문제 정의](#문제-정의)
- [접근 방법](#접근-방법)
- [프로젝트 구조](#프로젝트-구조)
- [실험 순서](#실험-순서)
- [결과 요약](#결과-요약)
- [팀원 소개](#팀원-소개)
- [내 역할](#내-역할)
- [트러블슈팅](#트러블슈팅)
- [한계점](#한계점)
- [문서 안내](#문서-안내)
- [재현 안내](#재현-안내)

## 프로젝트 개요

재활용품 이미지와 객관식 질문을 함께 입력받아 정답 선택지 `a/b/c/d`를 예측하는 Visual Question Answering(VQA) 대회 프로젝트입니다. Qwen 계열 VLM fine-tuning을 중심으로 모델 교체, 데이터 진단, 학습 파라미터 조정, 확률 기반 앙상블, detection/TTA 추론을 검증했습니다.

- 대회: SSAFY AI Challenge 2026
- 과제: 재활용품 이미지 기반 객관식 VQA
- 최종 제출 점수: 0.94402
- 리더보드: Private 12위, 팀명 `C044_동준이와 아이들`
- 상세 대회 설명: [docs/Overview.md](docs/Overview.md)

![Private leaderboard snapshot](Image/private_12th.png)

## 주요 정리

### 핵심 적용

최종 제출은 `Qwen3-VL-32B LoRA`와 `Qwen3.5-27B` 추론 결과를 활용하고, Grounding DINO·Florence-2 기반 crop, 좌우반전 TTA, a/b/c/d 확률 결합을 더한 파이프라인으로 구성했습니다. 이 흐름은 `notebooks/final_score_0.94402.ipynb`와 [detection/TTA 최종 제출 기록](docs/detection-tta-0.94402.md)에 정리했습니다.

### 추가 검증

초기 baseline 이후 `openbmb/MiniCPM-V-4_5`, `google/gemma-4-E4B-it`, `Qwen/Qwen3-VL-8B-Instruct`를 비교해 32B 확장의 기준을 잡았습니다. EDA에서는 dev 응답 불일치, 질문 유형, count/material 문제를 확인했고, 학습 단계에서는 learning rate, scheduler, label smoothing, LoRA 설정, answer 분포를 함께 보며 조정했습니다.

## 문제 정의

이 과제는 이미지와 질문을 동시에 이해해야 하는 객관식 VQA 문제입니다. 모델은 재질, 개수, 형태, 라벨, 포장 상태를 이미지에서 읽고, 질문 문맥에 맞는 선택지를 골라야 했습니다.

특히 dev 응답 불일치와 count/material 질문은 단순 fine-tuning 점수만으로 설명하기 어려웠습니다. 같은 이미지라도 질문 대상이 모호하거나, 객체가 겹치거나, 재질 판단에 시각적 단서가 부족한 경우가 있어 데이터 신뢰도와 샘플별 불확실성을 함께 봐야 했습니다.

- 입력: 재활용품 이미지 + 질문 + 보기 `a/b/c/d`
- 출력: 정답 선택지 한 글자
- 평가: 제출 CSV의 `id, answer` 기준 정확도
- 공개 데이터: CSV는 전체 행을 포함하고, 이미지는 `train/dev/test` split별 10장 샘플만 포함

## 접근 방법

![Pipeline](Image/pipeline.svg)

1. `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline으로 데이터 로드, 프롬프트 구성, 학습, 제출 CSV 생성을 먼저 검증했습니다.
2. 모델 카드와 baseline 구조를 기준으로 processor, model class, chat template, 입력 포맷을 바꾸며 후보 모델을 비교했습니다.
3. EDA로 dev 응답 불일치, 질문 유형, 정답 분포, count/material 문제를 확인했습니다.
4. 학습 로그와 제출 결과를 함께 보며 learning rate, scheduler, optimizer, label smoothing, LoRA 설정을 조정했습니다.
5. `Qwen/Qwen3-VL-32B-Instruct` LoRA 학습으로 확장하고, answer와 a/b/c/d 확률을 함께 저장했습니다.
6. Qwen3-VL-32B와 Qwen3.5-27B의 확률을 결합하는 soft ensemble을 검증했습니다.
7. 최종 제출에서는 detector crop, 원본 이미지, TTA, 듀얼 VQA 확률 결합을 사용했습니다.

## 프로젝트 구조

```text
ssafy-ai-challenge-2026-vqa/
├── README.md
├── Image/
│   ├── banner.png
│   ├── members.png
│   ├── private_12th.png
│   ├── pipeline.svg
│   └── score.svg
├── data/
│   ├── README.md
│   ├── train.csv / dev.csv / test.csv
│   ├── example_submission.csv
│   ├── train/
│   ├── dev/
│   └── test/
├── docs/
│   ├── README.md
│   ├── Overview.md
│   ├── Dataset Description.md
│   ├── detection-retrospective.md
│   ├── detection-tta-0.94402.md
│   ├── model-change-experiment.md
│   └── selective-thinking-failed-experiment.md
└── notebooks/
    ├── README.md
    ├── Baseline_0.70279.ipynb
    ├── final_score_0.94402.ipynb
    ├── train_0.9333.ipynb
    ├── ensemble_0.94363.ipynb
    └── experimental/
        ├── README.md
        └── EDA.ipynb
```

## 실험 순서

### 1. Baseline 확보

`Baseline_0.70279.ipynb`는 `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline입니다. 이미지 크기는 384로 고정했고, train 데이터 일부로 end-to-end 실행을 먼저 확인했습니다.

### 2. 모델 변경 실험

Baseline 코드의 구조를 유지한 채 Hugging Face 모델 카드에 맞춰 모델 ID, processor, model class, trust_remote_code, dtype, chat template을 바꿨습니다.

| 모델 | 점수 | 판단 |
|---|---:|---|
| `openbmb/MiniCPM-V-4_5` | 0.78341 | baseline보다 높았지만 Qwen3-VL-8B보다 낮음 |
| `google/gemma-4-E4B-it` | 0.82815 | 최신 모델이었지만 Qwen3-VL-8B보다 낮음 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | 세 후보 중 가장 높아 32B 확장의 기준으로 채택 |

### 3. EDA와 데이터 진단

`experimental/EDA.ipynb`에서 train/dev/test 구조, dev 응답 불일치, 질문 유형, 정답 분포, count/material 문제를 확인했습니다. 이 단계는 이후 학습 설정과 후반 추론 실험의 우선순위를 잡는 기준이 됐습니다.

### 4. 학습 파라미터 조정

초기 loss, 작은 샘플 overfit 여부, learning rate 후보, scheduler, label smoothing, optimizer, answer 분포를 같이 확인했습니다. MiniCPM-V-4_5에서 답이 한쪽으로 쏠리는 사례를 확인한 뒤, loss뿐 아니라 a/b/c/d 출력 안정성도 함께 봤습니다.

### 5. Qwen3-VL-32B LoRA 학습

`train_0.9333.ipynb`는 `Qwen/Qwen3-VL-32B-Instruct`를 Unsloth와 LoRA로 학습한 노트북입니다. 결과는 0.93141이며, 이 단계부터 answer와 a/b/c/d 확률 저장 구조를 포함했습니다.

### 6. Soft ensemble

`ensemble_0.94363.ipynb`는 Qwen3-VL-32B 추론 결과와 Qwen3.5-27B 추론 결과의 a/b/c/d 확률을 가중 합산한 실험입니다. 결과는 0.94363입니다.

### 7. Detection / TTA 최종 제출

`final_score_0.94402.ipynb`는 Grounding DINO와 Florence-2로 관심 객체 crop을 만들고, 원본 이미지와 crop 이미지를 함께 넣어 VQA 추론한 뒤 TTA와 듀얼 모델 확률을 결합한 최종 제출 파이프라인입니다.

## 결과 요약

![Score](Image/score.svg)

| 구분 | 결과 | 설명 |
|---|---:|---|
| 최종 제출 | 0.94402 | `final_score_0.94402.ipynb` 최종 제출 파이프라인 |
| 리더보드 | Private 12위 | `Image/private_12th.png`에 결과 캡처 보관 |
| Baseline | 0.70279 | Qwen2.5-VL-3B-Instruct 4bit LoRA baseline |
| 모델 변경 | 0.78341 / 0.82815 / 0.83643 | MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B 비교 |
| 32B LoRA | 0.93141 | Qwen3-VL-32B LoRA 학습과 확률 저장 구조 |
| Soft ensemble | 0.94363 | Qwen3-VL-32B와 Qwen3.5-27B 확률 결합 |
| Detection / TTA | 0.94402 | crop + TTA + 듀얼 VQA 확률 결합 |

## 팀원 소개

![팀원 소개](Image/members.png)

- 장동준: 팀장. 팀 채널 운영과 제출 흐름을 맡았고, 27B/35B 계열 시도를 진행했습니다.
- 정지성: 후반 추론 파이프라인과 앙상블·TTA·detection 계열 실험을 주로 진행했습니다.
- 조용주: 이전 대회 코드 조사, baseline 개선 아이디어, 실험 비교 자료 공유를 맡았습니다.
- 이진영: 파라미터 수정 실험과 토큰 처리 이슈 점검을 맡았고, all-a 붕괴 사례를 팀에 공유했습니다.
- 김동혁: 아래 `내 역할` 섹션에서 따로 정리했습니다.

## 내 역할

| 영역 | 수행 내용 | 관련 파일 |
|---|---|---|
| 데이터 진단 | dev 응답 불일치, 질문 유형, 정답 분포, count/material 문제를 분석하고 후속 실험 기준을 잡음 | `notebooks/experimental/EDA.ipynb` |
| 모델 후보 검증 | baseline 구조를 바탕으로 MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B를 실행하고 점수와 출력 안정성을 비교 | `docs/model-change-experiment.md` |
| 학습 설정 조정 | learning rate, scheduler, label smoothing, optimizer, LoRA 설정을 검토하고 answer 분포까지 함께 확인 | `docs/model-change-experiment.md`, `notebooks/train_0.9333.ipynb` |
| 객체 인식 방향 제안 | Qwen3-VL-32B가 흔들리는 count/material 샘플을 보고, 질문 대상 객체를 먼저 좁히는 detection 방향을 제안 | `docs/detection-retrospective.md` |

## 트러블슈팅

### 출력 안정성 문제

객관식 VQA에서는 자연스러운 문장 생성보다 `a/b/c/d` 한 글자를 안정적으로 내는 것이 중요했습니다. MiniCPM-V-4_5 실험에서 답이 한쪽으로 쏠리는 현상을 확인했고, 이후 실험에서는 loss와 함께 answer 분포를 같이 확인했습니다.

### 모델 선택 문제

`google/gemma-4-E4B-it`는 `Qwen/Qwen3-VL-8B-Instruct`보다 낮은 점수를 기록했습니다. 이후 실험은 대회 포맷, chat template, 출력 파싱이 안정적인 Qwen3-VL 기준으로 이어갔습니다.

### 어려운 샘플 처리 문제

개수와 재질 질문은 원본 이미지 한 장만으로 불안정한 경우가 있었습니다. 후반에는 관심 객체 crop, 원본 이미지, TTA, 듀얼 VQA 확률 결합으로 보완했습니다.

## 한계점

- 객체 인식은 대회 후반 추론 보조로 붙인 방식이라, 학습 데이터 구성이나 모델 입력 구조에 초반부터 통합하지 못했습니다.
- 사람도 판단하기 어려운 개수 문제와 재질 문제는 crop을 붙여도 완전히 해결되지 않았습니다.
- 배경이 복잡하거나 객체가 겹친 장면에서는 detector crop 자체가 충분히 깔끔하지 않은 경우가 있었습니다.
- selective Thinking처럼 어려운 샘플만 다시 보는 후처리 아이디어는 로그와 최종 answer 연결이 불안정해 최종 제출 흐름에 포함하지 않았습니다.
- 개선 방향은 질문 대상 객체 탐지, segmentation, 샘플별 불확실성 기반 routing을 학습 초반부터 설계하는 것입니다.

## 문서 안내

- [대회 개요](docs/Overview.md)
- [Dataset Description](docs/Dataset%20Description.md)
- [노트북 설명](notebooks/README.md)
- [모델 변경 실험 기록](docs/model-change-experiment.md)
- [객체 인식 회고](docs/detection-retrospective.md)
- [detection/TTA 최종 제출 기록](docs/detection-tta-0.94402.md)
- [selective Thinking 실패 실험 기록](docs/selective-thinking-failed-experiment.md)
- [데이터 샘플 구조](data/README.md)

## 재현 안내

학습된 모델 weight와 checkpoint는 포함하지 않습니다. 공개 노트북은 실험 순서와 주요 코드를 보여주기 위한 파일입니다. `data/`에는 전체 CSV와 split별 이미지 샘플 10장씩을 포함했습니다. 이 저장소에서는 모델명, 학습 코드, 추론 로직, 확인된 제출 점수를 중심으로 재현 범위를 정리했습니다.

## License

이 저장소는 SSAFY AI Challenge 수행 내용을 포트폴리오/회고 목적으로 구성한 공개용 저장소입니다. 코드와 문서의 재사용 범위는 별도 `LICENSE` 파일을 추가해 명시하는 것을 권장합니다.
