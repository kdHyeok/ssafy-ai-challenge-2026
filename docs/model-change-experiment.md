# Multimodal VQA 모델 변경 실험 기록

이 문서는 baseline 코드를 출발점으로 모델만 바꾸며 실행한 초기 실험을 정리합니다. 목적은 모든 하이퍼파라미터를 완전히 통제한 ablation이 아니라, 어떤 모델이 이 대회의 객관식 VQA 포맷에 더 잘 맞는지 빠르게 확인하는 것이었습니다.

## 점수 요약

| Stage | 모델 | 실제 점수 | 변화 |
|---|---|---:|---:|
| Baseline | `Qwen/Qwen2.5-VL-3B-Instruct` | 0.70279 | - |
| Model change 1 | `openbmb/MiniCPM-V-4_5` | 0.78341 | +0.08062 |
| Model change 2 | `google/gemma-4-E4B-it` | 0.82815 | +0.12536 |
| Model change 3 | `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | +0.13364 |

`google/gemma-4-E4B-it`는 최신 모델이었지만 실제 점수는 `Qwen/Qwen3-VL-8B-Instruct`보다 낮았습니다. 그래서 이후 실험은 Qwen3-VL-8B를 기준으로 잡고 같은 구조의 32B 모델로 넘어갔습니다.

## Stage 1 — Baseline: Qwen2.5-VL-3B-Instruct

관련 노트북: `../notebooks/Baseline_0.70279.ipynb`

Baseline은 `Qwen/Qwen2.5-VL-3B-Instruct`에 4bit quantization과 LoRA를 붙여 학습하는 구조였습니다. train 200개 샘플과 384 고정 해상도로 빠르게 end-to-end 실행을 확인하는 역할에 가깝습니다.

## Stage 2 — MiniCPM-V-4_5: 0.78341

실험 모델:

```text
openbmb/MiniCPM-V-4_5
```

MiniCPM-V-4_5는 baseline보다 높은 점수를 기록했습니다. 다만 모델 카드 기준으로 `AutoModel`과 `trust_remote_code=True`를 써야 했고, chat 입력 형식과 출력 파싱도 따로 맞춰야 했습니다.

## Stage 3 — Gemma 4 E4B: 0.82815

실험 모델:

```text
google/gemma-4-E4B-it
```

Gemma는 Qwen 계열과 chat template이 달라서 객관식 a/b/c/d 출력 안정화를 따로 잡아야 했습니다. 최신 모델이었지만 점수는 Qwen3-VL-8B보다 낮았습니다.

## Stage 4 — Qwen3-VL-8B-Instruct: 0.83643

실험 모델:

```text
Qwen/Qwen3-VL-8B-Instruct
```

Qwen2.5-VL baseline과 같은 계열이라 프롬프트 구조와 a/b/c/d 출력 파싱을 재사용하기 쉬웠고, 실제 점수도 세 후보 중 가장 높았습니다. 이 결과 때문에 이후에는 Qwen3-VL-8B를 기준 모델로 삼고 `Qwen/Qwen3-VL-32B-Instruct`로 확장했습니다.

## 학습 파라미터 조정 근거

학습 파라미터는 수업 메모와 직접 비교 실험을 바탕으로 정리했습니다.

수업 메모 쪽에서 참고한 판단 기준:
1. 초기 loss가 class 수 기준 기대값과 비슷한지 본다.
2. 작은 샘플을 빠르게 학습시켜 loss가 줄어드는지 본다.
3. learning rate 후보를 짧게 비교해 발산하거나 너무 느린 조합을 뺀다.
4. train loss와 validation 결과를 분리해서 본다.
5. scheduler는 조합을 본 뒤 적용한다.
6. loss 이동 평균과 validation loss를 같이 본다.

직접 비교 실험에서 체크한 조합:

```text
EPOCHS = 3
LEARNING_RATE = 3e-5
LR_SCHEDULER = constant_with_warmup 또는 cosine
LOSS_TYPE = label_smoothing 0.1
OPTIMIZER = adamw_torch
```

MiniCPM-V-4_5에서는 특정 `constant_with_warmup` 설정에서 답변이 a로 고정되는 현상도 확인했습니다. 이 경험 때문에 loss만이 아니라 answer 분포와 a/b/c/d 출력 안정성도 같이 봤습니다.

32B 학습에서는 아래 설정을 기준으로 갔습니다.

```python
learning_rate=1e-4
lr_scheduler_type="cosine"
per_device_train_batch_size=24
gradient_accumulation_steps=1
num_train_epochs=5
weight_decay=0.01
LoRA r=32
lora_alpha=32
```

후속 비교에서는 `learning_rate=5e-5`, batch size 16, `lora_alpha=16`, warmup 비율 0.10도 검토했습니다.

## 실험에서 배운 점

1. 모델 크기만이 아니라 태스크 적합성과 출력 안정성이 중요했습니다.
2. 최신 모델이 항상 최고점으로 이어지지는 않았습니다.
3. 객관식 VQA에서는 생성 품질보다 a/b/c/d 한 글자 출력 안정성이 중요했습니다.
4. 학습 파라미터는 loss만 보고 고르지 않았고, answer 분포와 실제 제출 결과를 같이 봤습니다.
5. Qwen3-VL-8B가 가장 잘 맞았기 때문에 32B 확장이 자연스럽게 이어졌습니다.
