# SSAFY AI Challenge 2026 VQA

![Banner](Image/banner.png)

재활용품 이미지와 객관식 질문을 입력으로 받아 정답 선택지를 예측하는 VQA 대회 프로젝트입니다. Baseline 코드를 출발점으로 모델 교체, 학습 파라미터 조정, Qwen3-VL-32B LoRA 학습, 확률 기반 추론 저장, 선택적 Thinking 재추론, soft ensemble, detection/TTA를 단계적으로 실험했습니다. 최고 제출 점수는 0.94402, Private Leaderboard 순위는 12위입니다.

## 주요 성과

- 최고 제출 점수: 0.94402
- 리더보드: Private Leaderboard 12위, 팀명 `C044_동준이와 아이들`
- 핵심 실험: Qwen3-VL-32B-Instruct LoRA 학습, a/b/c/d logit 기반 확률 저장, confidence 기반 선택적 재추론, Qwen3-VL-32B-Thinking 후처리
- 모델 변경 실험: `openbmb/MiniCPM-V-4_5` 0.78341, `google/gemma-4-E4B-it` 0.82815, `Qwen/Qwen3-VL-8B-Instruct` 0.83643
- 기타 실험: MiniCPM-V-4.6 검토, 학습 파라미터 조정, Qwen3.5-27B soft ensemble, detection/TTA 후반 확장 실험

![Private leaderboard snapshot](Image/private_12th.png)

## Team Members

![Team Members](Image/members.png)

- 장동준: Team Leader
- 정지성: Member
- 조용주: Member
- 김동혁: Member
- 이진영: Member

## 문제 정의

- 입력: 재활용품 이미지 + 질문 + 보기 a/b/c/d
- 출력: 정답 선택지 한 글자
- 평가: 제출 CSV의 `id, answer` 기준 정확도
- 데이터 샘플: `data/`에 train/dev/test split별 10행 CSV와 이미지 샘플 10장씩 포함

주요 난점은 이미지와 텍스트를 동시에 이해해야 한다는 점입니다. 모델은 재질, 개수, 형태, 라벨, 포장 상태처럼 시각적 세부 정보를 질문 문맥과 함께 판단해야 했습니다. 또한 dev 응답 불일치와 count/material 질문은 단순 fine-tuning 점수만으로 해석하기 어려워, 데이터 신뢰도와 샘플별 불확실성을 함께 보는 방식으로 접근했습니다.

## 접근 방법 요약

![Pipeline](Image/pipeline.svg)

1. `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline으로 데이터 로드, 프롬프트 구성, 학습, 제출 CSV 생성까지 end-to-end 흐름을 확인했습니다.
2. Hugging Face 모델 카드와 baseline 코드를 기준으로 모델 ID, processor, model class, chat template, 입력 포맷을 바꾸며 `openbmb/MiniCPM-V-4_5`, `google/gemma-4-E4B-it`, `Qwen/Qwen3-VL-8B-Instruct`를 실행했습니다.
3. 모델 크기가 커질수록 점수는 전반적으로 상승했습니다. 다만 최신 모델인 `google/gemma-4-E4B-it`는 0.82815로 `Qwen/Qwen3-VL-8B-Instruct`의 0.83643보다 낮았습니다. 출시 직후라 레퍼런스와 튜닝 사례가 부족하다고 판단해 최종 확장 방향은 Qwen3-VL-8B로 잡았습니다.
4. 수업 메모의 학습 점검 흐름과 직접 비교 실험을 바탕으로 learning rate, scheduler, optimizer, label smoothing, LoRA rank/alpha, batch size를 조정했습니다. 관련 그래프 이미지는 추후 추가할 수 있도록 별도 실험 문서에 자리를 남겼습니다.
5. Qwen3-VL-32B-Instruct LoRA 학습으로 확장하고, answer뿐 아니라 a/b/c/d 확률을 저장해 선택적 재추론과 soft ensemble 비교가 가능하도록 했습니다.
6. 여러 제출에서 답이 갈리거나 confidence가 낮은 샘플을 대상으로 Qwen3-VL-32B-Thinking 재추론을 적용했습니다.
7. 후반에는 Qwen3.5-27B soft ensemble과 공개 detector 기반 detection/TTA 보조 실험을 비교했습니다.

자세한 모델 교체 과정은 [모델 변경 실험 기록](notebooks/model-change-experiment.md)에 정리했습니다.

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
│   ├── sample_submission_format.csv
│   ├── train/
│   ├── dev/
│   └── test/
└── notebooks/
    ├── README.md
    ├── model-change-experiment.md
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

### 1. Baseline 확보

`Baseline_0.70279.ipynb`는 `Qwen/Qwen2.5-VL-3B-Instruct` 기반 4bit LoRA baseline입니다. 이미지 크기는 384로 고정했고, train 데이터 200개를 샘플링해 1차 end-to-end 흐름을 검증했습니다. 이 단계의 목적은 최고점을 만드는 것이 아니라 데이터 로드, 이미지-질문-보기 프롬프트 구성, 학습 루프, 제출 CSV 생성이 정상적으로 이어지는지 확인하는 것이었습니다.

### 2. 모델 변경 실험

Baseline 코드의 구조를 유지한 채 Hugging Face 모델 카드에 맞춰 모델 ID, processor, model class, trust_remote_code, dtype, chat template을 바꾸며 후보 모델을 실행했습니다.

| 모델 | 기록 점수 | 해석 |
|---|---:|---|
| `openbmb/MiniCPM-V-4_5` | 0.78341 | baseline보다 높았지만 Qwen3-VL-8B보다 낮음 |
| `google/gemma-4-E4B-it` | 0.82815 | 최신 모델이었지만 출시 당일 레퍼런스 부족과 VQA 출력 포맷 안정화 문제를 고려함 |
| `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | 세 후보 중 가장 높았고 Qwen3-VL-32B 확장의 기준 모델로 채택 |

이 결과만 보면 모델 크기가 커질수록 점수가 전반적으로 상승했습니다. 다만 `google/gemma-4-E4B-it`는 최신 모델이라는 장점에도 `Qwen/Qwen3-VL-8B-Instruct`보다 낮았기 때문에, 대회 기간 안에서는 레퍼런스가 더 안정적인 Qwen3-VL을 선택했습니다.

### 3. 학습 파라미터 조정

학습 파라미터는 두 자료를 기준으로 정리했습니다.

- 수업 메모: loss 초기값 확인, 작은 샘플 과적합 테스트, learning rate 후보 탐색, train/validation 분리, scheduler 적용 순서
- 직접 비교 실험: Qwen3-VL-8B sample-50 실험에서 `3e-5`, `constant_with_warmup`, `cosine`, `label_smoothing=0.1`, `adamw_torch` 비교

이후 32B 학습에서는 `learning_rate=1e-4`, `lr_scheduler_type="cosine"`, `per_device_train_batch_size=24`, `gradient_accumulation_steps=1`, `num_train_epochs=5`, `LoRA r=32`, `lora_alpha=32`, `weight_decay=0.01` 설정을 사용했습니다. 후속 비교에서 batch size, learning rate, LoRA alpha, warmup 비율을 조정하며 메모리 제약과 validation 흐름을 함께 봤습니다.

### 4. Qwen3-VL-32B LoRA 학습

`train_0.9333.ipynb`는 `Qwen/Qwen3-VL-32B-Instruct`를 Unsloth와 LoRA로 학습한 대표 노트북입니다. 8B 실험에서 Qwen3-VL이 가장 높은 점수를 기록했기 때문에 같은 Qwen3-VL 구조에서 32B로 확장했습니다.

이 단계부터 단순히 answer만 저장하는 방식에서 벗어나 a/b/c/d 확률을 함께 저장했습니다. 이 구조는 이후 선택적 Thinking 재추론과 soft ensemble의 입력 자료가 되었습니다.

### 5. 선택적 Thinking 재추론

`score_0.94402.ipynb`는 여러 제출 결과가 일치하는 쉬운 샘플과 답이 갈리는 저합의 샘플을 나누고, 저합의 샘플에 `Qwen/Qwen3-VL-32B-Thinking`을 적용하는 후처리 실험입니다.

이 방식은 최고점 0.94402 제출과 연결됩니다. 다만 Thinking만의 독립 기여량을 별도 ablation으로 분리하지 않았고, 모든 샘플에 Thinking을 적용한 전략이 아니라 낮은 confidence와 answer disagreement가 있는 샘플만 보정한 전략으로 정리합니다.

### 6. Soft ensemble과 confidence routing

`ensemble_0.94363.ipynb`는 Qwen3-VL-32B 추론 결과를 먼저 만들고, confidence가 낮은 샘플에 Qwen3.5-27B를 보조 모델로 적용한 뒤 a/b/c/d 확률을 가중 결합하는 soft ensemble 실험입니다.

이 실험은 0.94363을 기록했지만 최고점 0.94402를 넘지는 못했습니다. 그래서 최종 핵심 성과라기보다, 확률 저장 구조와 confidence routing을 실제로 활용한 후반 비교 실험으로 정리합니다.

### 7. Detection/TTA와 Falcon 회고

후반에는 count, 작은 물체, 객체 인식 문제를 보완하기 위해 detection crop과 TTA를 붙이는 실험을 진행했습니다. `detection_tta_0.93102.ipynb`는 Grounding DINO와 Florence-2 공개 detector를 별도 학습 없이 사용해 관심 객체 후보 crop을 만들고, 원본 이미지와 crop 이미지를 VQA 추론 입력에 함께 활용한 실험입니다.

이 실험은 0.93102를 기록했고, 복잡도 대비 최고점 개선으로 이어지지는 않았습니다. Falcon-Perception은 최종 제출 파이프라인에 포함하지 않았고, object tag, crop routing, pseudo-label 보정, VQA 학습 입력으로 초반부터 연결했으면 좋았겠다는 후속 개선 아이디어로 정리했습니다.

## 대표 결과

![Score](Image/score.svg)

| 구분 | 대표 결과 | 해석 |
|---|---:|---|
| 최고 제출 | 0.94402 | Qwen3-VL-32B 기반 추론과 선택적 Thinking 재추론을 결합한 결과 |
| 리더보드 | Private 12위 / 0.94402 | `Image/private_12th.png`에 보관된 결과 캡처 |
| Baseline | 0.70279 | Qwen2.5-VL-3B-Instruct 4bit LoRA baseline |
| 모델 변경 | 0.78341 → 0.82815 → 0.83643 | MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B 순서로 비교 |
| Qwen3-VL-32B 학습 | 0.9333 | 32B LoRA 학습과 a/b/c/d 확률 저장 구조 확보 |
| Soft ensemble | 0.94363 | Qwen3.5-27B와 확률 기반 가중 결합을 시도했지만 최고점은 넘지 못함 |
| Detection/TTA | 0.93102 | 공개 detector 기반 crop/TTA 보조 실험. 최고점 개선으로 이어지지는 않음 |

## 내가 기여한 부분

팀 프로젝트에서 저는 데이터 분석, 초기 모델 변경 실험, 학습 파라미터 비교, Qwen3-VL-32B 확장, 확률 기반 추론 저장 구조, 선택적 재추론 방향 정리에 기여했습니다. 최종 점수는 팀의 반복 실험 결과로 보고, 아래에는 제가 직접 작성하거나 제안한 작업을 중심으로 정리했습니다.

| 영역 | 기여 내용 | 산출물 |
|---|---|---|
| EDA 및 데이터 분석 | dev 응답 불일치, 질문 유형, count/material 문제를 분석하고 후속 실험 우선순위를 정리 | `notebooks/experimental/EDA.ipynb` |
| 모델 변경 실험 | baseline 코드에서 모델 카드에 맞춰 MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B를 실행하고 Qwen3-VL-8B를 채택 | `notebooks/model-change-experiment.md` |
| 학습 파라미터 조정 | 수업 메모와 직접 비교 실험을 바탕으로 learning rate, scheduler, label smoothing, optimizer, LoRA 설정을 비교 | `notebooks/model-change-experiment.md`, `notebooks/train_0.9333.ipynb` |
| 32B 학습 실험 | Qwen3-VL-32B LoRA 학습 구성과 pseudo-label, augmentation, 확률 저장 흐름을 정리 | `notebooks/train_0.9333.ipynb` |
| 추론 결과 저장 구조 | answer만 저장하지 않고 a/b/c/d 확률을 함께 저장하는 포맷을 제안해 ensemble과 재추론 비교가 가능하도록 함 | `notebooks/score_0.94402.ipynb`, `notebooks/ensemble_0.94363.ipynb` |
| 선택적 재추론 | answer disagreement와 낮은 confidence 샘플을 대상으로 Thinking 재추론을 적용하는 방향을 제안 | `notebooks/score_0.94402.ipynb` |
| 실패 실험 정리 | detection/TTA 등 최고점으로 이어지지 않은 후반 실험도 점수와 한계를 분리해 정리 | `notebooks/experimental/detection_tta_0.93102.ipynb` |

## 주요 트러블슈팅과 배운 점

### 모델 최신성과 태스크 적합성

최신 모델이 항상 대회 점수로 바로 이어지지는 않았습니다. `google/gemma-4-E4B-it`는 대회 당일 출시된 최신 모델이었지만 0.82815로 `Qwen/Qwen3-VL-8B-Instruct`의 0.83643보다 낮았습니다. 출시 직후라 예제와 레퍼런스가 부족했고, 객관식 VQA에서 중요한 프롬프트, chat template, a/b/c/d 출력 파싱을 충분히 안정화하기 어려웠습니다. 그래서 최신성보다 대회 포맷에서의 실제 점수와 튜닝 안정성을 기준으로 Qwen3-VL-8B를 채택했습니다.

### 학습 파라미터 탐색

수업 메모에서 정리한 것처럼 초기 loss, 작은 샘플 학습, learning rate 후보, train/validation 흐름을 먼저 확인했습니다. 직접 비교 실험에서는 scheduler와 label smoothing 조합에 따라 loss 흐름이 크게 달라졌고, MiniCPM-V-4_5에서는 특정 설정에서 답변이 a로 고정되는 현상도 확인했습니다. 이 경험을 바탕으로 32B 학습에서는 cosine scheduler, AdamW, weight decay, LoRA rank/alpha, batch size를 더 신중하게 조정했습니다.

### GPU 자원 제약

32B급 VLM은 성능 가능성이 있었지만, 학습 시간이 길고 OOM이 쉽게 발생했습니다. 모델을 무리하게 크게만 쓰기보다 4bit 양자화와 LoRA를 기준으로 실험 가능성을 먼저 확보했고, batch size, gradient accumulation, 학습 스케줄을 조정하며 반복 가능한 실험 단위를 만들었습니다.

### 후처리 구조와 선택적 재추론

제출 파일에 최종 answer만 남기면 모델들이 어디서 흔들리는지 비교하거나 앙상블을 만들기 어려웠습니다. 그래서 a/b/c/d 확률을 함께 저장하는 포맷을 제안했고, 이를 바탕으로 confidence가 낮거나 제출 간 답이 갈리는 샘플을 따로 보는 흐름을 만들었습니다.

### VLM의 객체 인식 한계

대회 막바지에는 일반 VLM이 재활용품 객체를 세밀하게 인식하지 못하는 문제가 남아 있었습니다. detection/TTA는 공개 detector로 crop 후보를 만들어 추론 입력에 보조로 붙인 방식이었고, 객체 인식 결과를 학습 데이터나 모델 앞단에 통합하지는 못했습니다. 다음에는 객체 인식 결과를 초반부터 학습 입력이나 샘플 분기 기준으로 활용해야 한다는 교훈을 얻었습니다.

## 성장한 점

- VLM fine-tuning 프로젝트를 baseline, EDA, 학습, 추론, 후처리, 제출, 회고까지 end-to-end로 경험했습니다.
- 모델 교체 실험에서 Hugging Face 모델 카드와 실제 baseline 코드를 연결해 processor, model class, chat template, dtype, 입력 포맷을 맞추는 과정을 경험했습니다.
- 32B급 모델을 제한된 GPU 환경에서 실행하기 위해 LoRA/QLoRA, 4bit quantization, Unsloth 최적화를 적용했습니다.
- 데이터 품질과 dev 활용 방식이 모델 성능만큼 중요하다는 것을 체감했습니다.
- 고득점 결과만 보여주는 것이 아니라, 효과가 제한적인 실험과 한계를 분리해 설명하는 포트폴리오 서사의 중요성을 배웠습니다.

## 문서 안내

- [대표 노트북 설명](notebooks/README.md)
- [모델 변경 실험 기록](notebooks/model-change-experiment.md)
- [데이터 샘플 구조](data/README.md)

## 재현 안내

학습된 모델 weight와 checkpoint는 포함하지 않습니다. 공개 노트북은 실험 흐름과 핵심 로직을 보여주기 위한 대표 파일입니다. `data/`에는 CSV 구조와 이미지 경로를 설명하기 위한 split별 10행 샘플과 이미지 샘플만 두었습니다. 공개 레포만으로 exact score 재현을 보장하지는 않지만, 대표 모델명, 핵심 학습 코드, 추론 로직, 점수 흐름을 검토할 수 있게 정리했습니다.

노트북은 Colab과 GPU 환경에서 수행된 대표 실험 기록이며, 각 노트북의 설치 셀과 실행 환경에 의존합니다. 단일 패키지 목록만으로 전체 대회 환경을 재현할 수 있다고 오해되지 않도록, 공개 루트에는 별도 환경 파일을 두지 않았습니다.

## License

이 저장소는 SSAFY AI Challenge 수행 내용을 포트폴리오/회고 목적으로 정리한 공개용 저장소입니다. 코드와 문서의 재사용 범위는 별도 `LICENSE` 파일을 추가해 명확히 정리하는 것을 권장합니다.
