# Multimodal VQA 성능 개선 기록: Baseline에서 Qwen3-VL-8B까지

이 문서는 baseline 코드를 출발점으로 모델만 바꾸며 실행한 초기 모델 변경 실험을 기록합니다. 목적은 모든 하이퍼파라미터를 완전히 통제한 ablation이 아니라, Hugging Face 모델 카드와 baseline 코드를 기준으로 모델을 교체했을 때 어떤 모델이 대회 객관식 VQA 포맷에 더 잘 맞는지 빠르게 확인하는 것이었습니다.

## 점수 요약

| Stage | 모델 | 점수 | 변화 |
|---|---|---:|---:|
| Baseline | `Qwen/Qwen2.5-VL-3B-Instruct` | 0.70279 | - |
| Model change 1 | `openbmb/MiniCPM-V-4_5` | 0.78341 | +0.08062 |
| Model change 2 | `google/gemma-4-E4B-it` | 0.82815 | +0.12536 |
| Model change 3 | `Qwen/Qwen3-VL-8B-Instruct` | 0.83643 | +0.13364 |

모델 크기와 VQA 적합성이 좋아질수록 점수는 올랐습니다. `google/gemma-4-E4B-it`는 최신 모델이었지만 대회 당일 출시 모델이라 레퍼런스와 튜닝 사례가 부족했고, 실제 점수도 `Qwen/Qwen3-VL-8B-Instruct`보다 근소하게 낮았습니다. 그래서 이후 실험은 Qwen3-VL-8B를 기준으로 잡고, 같은 Qwen3-VL 구조의 32B 모델로 넘어갔습니다.

## Stage 1 — Baseline: Qwen2.5-VL-3B-Instruct

관련 노트북: `Baseline_0.70279.ipynb`

Baseline은 `Qwen/Qwen2.5-VL-3B-Instruct`에 4bit quantization과 LoRA를 붙여 학습하는 구조였습니다.

```python
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_SIZE = 384

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=IMAGE_SIZE * IMAGE_SIZE,
    max_pixels=IMAGE_SIZE * IMAGE_SIZE,
    trust_remote_code=True,
)

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

이 baseline은 학습 데이터 200개만 사용하고, 이미지를 384 고정 해상도로 처리했습니다. 이 설정은 빠르게 end-to-end 실행을 확인하기에는 좋았지만, 전체 데이터 다양성과 Qwen-VL의 동적 해상도를 쓰지 못했습니다.

## Stage 2 — MiniCPM-V-4_5: 0.78341

실험 모델:

```text
openbmb/MiniCPM-V-4_5
```

MiniCPM-V-4_5 모델 카드는 Qwen3-8B와 SigLIP2-400M 기반의 멀티모달 모델이며, 단일 이미지, 다중 이미지, 비디오 이해, OCR, document parsing을 내세웁니다.

Baseline 코드에서 바꾼 내용은 다음과 같습니다.

```python
MODEL_ID = "openbmb/MiniCPM-V-4_5"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

Qwen2.5-VL 전용 `Qwen2_5_VLForConditionalGeneration` class에 그대로 넣는 방식이 아니라, 모델 카드의 예시처럼 `AutoModel`과 `trust_remote_code=True`를 기준으로 맞춰야 했습니다. 또한 chat 입력 형식과 이미지 전달 방식이 Qwen baseline과 달라서, baseline의 프롬프트 구성과 출력 파싱을 MiniCPM 방식에 맞게 조정해야 했습니다.

결과는 0.78341이었습니다. Baseline보다 높았지만, 이후 Gemma 4 E4B와 Qwen3-VL-8B보다 낮았습니다.

## Stage 3 — Gemma 4 E4B: 0.82815

실험 모델:

```text
google/gemma-4-E4B-it
```

Gemma 4 E4B는 instruction-tuned E4B 모델입니다. 모델 카드 기준으로 E4B는 4.5B effective parameters, 8B total parameters with embeddings이며, 이미지와 텍스트 입력을 지원합니다.

Baseline 코드에서 바꾼 내용은 다음과 같습니다.

```python
MODEL_ID = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

Gemma는 Qwen chat template과 토큰 구성이 다르기 때문에, baseline의 Qwen 전용 message format을 그대로 쓰면 a/b/c/d 한 글자 출력이 흔들릴 수 있습니다. 객관식 VQA에서는 모델 성능뿐 아니라 최종 출력이 정확히 한 글자로 떨어지는지도 중요하므로, system/user message와 answer parsing을 Gemma 방식에 맞게 정리해야 했습니다.

결과는 0.82815였습니다. 최신 모델이었지만 `Qwen/Qwen3-VL-8B-Instruct`보다 낮았습니다. 면접에서 이 질문을 받는다면 다음처럼 설명할 수 있습니다.

> 최신 모델이라고 해서 대회 점수가 바로 높아지지는 않았습니다. Gemma 4 E4B는 대회 당일 출시된 모델이라 레퍼런스와 튜닝 사례가 부족했고, 객관식 VQA에서 필요한 chat template, 프롬프트, a/b/c/d 출력 파싱을 다듬기 어려웠습니다. 실제 점수도 Qwen3-VL-8B보다 낮았기 때문에, 대회 기간 안에서는 더 안정적으로 점수가 나온 Qwen3-VL-8B를 채택했습니다.

## Stage 4 — Qwen3-VL-8B-Instruct: 0.83643

실험 모델:

```text
Qwen/Qwen3-VL-8B-Instruct
```

Qwen3-VL-8B 모델 카드는 visual perception, spatial understanding, OCR, multimodal reasoning, long context를 강조합니다. Hugging Face 예시는 `Qwen3VLForConditionalGeneration`과 `AutoProcessor` 사용을 안내합니다.

Baseline 코드에서 바꾼 내용은 다음과 같습니다.

```python
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
```

Qwen2.5-VL baseline과 같은 Qwen3-VL 사용 방식이라 프롬프트 구조와 a/b/c/d 출력 파싱을 재사용하기 쉬웠습니다. 또한 Hugging Face 모델 카드의 예제가 비교적 명확했고, 실제 점수도 0.83643으로 세 모델 중 가장 높았습니다.

이 결과 때문에 이후에는 Qwen3-VL-8B를 기준 모델로 삼고, 같은 Qwen3-VL 구조에서 `Qwen/Qwen3-VL-32B-Instruct`로 확장했습니다.

## 학습 파라미터 조정 근거

학습 파라미터는 두 자료를 참고했습니다.

- 수업 메모: https://woolly-lint-d2a.notion.site/AI_2_-33410fe5c97b8008886dca7043c9cd2d?source=copy_link
- 직접 비교 실험: https://woolly-lint-d2a.notion.site/AI-33710fe5c97b8060a3c7d0bfbc6964d3?source=copy_link

수업 메모에서 가져온 판단 기준은 다음과 같습니다.

1. 초기 loss가 class 수 기준 기대값과 비슷한지 확인한다.
2. 작은 샘플을 빠르게 학습시켜 loss가 줄어드는지 확인한다.
3. learning rate 후보를 짧게 비교해 발산하거나 너무 느린 조합을 제거한다.
4. train loss와 validation performance를 분리해서 본다.
5. scheduler는 조합 탐색 후 적용한다.
6. loss 이동평균과 validation loss을 같이 확인한다.

직접 비교 실험에서는 Qwen3-VL-8B sample-50 학습에서 다음 조합을 확인했습니다.

```text
EPOCHS = 3
LEARNING_RATE = 3e-5
LR_SCHEDULER = constant_with_warmup 또는 cosine
LOSS_TYPE = label_smoothing 0.1
OPTIMIZER = adamw_torch
```

MiniCPM-V-4_5에서는 특정 `constant_with_warmup` 설정에서 답변이 a로 고정되는 현상도 확인했습니다. 이 경험 때문에 단순히 loss만 보지 않고, 실제 제출 answer 분포와 a/b/c/d 출력 안정성도 함께 확인했습니다.

32B 학습에서는 다음 설정을 사용했습니다.

```python
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

training_args = SFTConfig(
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=24,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    weight_decay=0.01,
)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
)
```

후속 비교에서는 768 리사이징, 이미지 변환 증강 제거, `learning_rate=5e-5`, batch size 16, `lora_alpha=16`, warmup 비율 0.10도 검토했습니다.

그래프 이미지는 추후 이 문서의 아래 위치에 추가할 예정입니다.

```markdown
![학습 파라미터 비교 그래프](../Image/training-parameter-comparison.png)
```

## 실험에서 배운 점

1. 모델 크기는 점수에 영향을 줬습니다. Qwen2.5-VL-3B baseline에서 MiniCPM-V-4_5, Gemma 4 E4B, Qwen3-VL-8B로 바꾸면서 점수가 0.70279 → 0.78341 → 0.82815 → 0.83643으로 올랐습니다.
2. 최신 모델이 항상 가장 좋은 선택은 아니었습니다. Gemma 4 E4B는 최신 모델이었지만 레퍼런스 부족과 출력 안정화 문제 때문에 Qwen3-VL-8B보다 낮았습니다.
3. 객관식 VQA에서는 생성 문장 품질보다 a/b/c/d 한 글자 출력 안정성이 중요했습니다.
4. 학습 파라미터는 loss만 보고 고르지 않았습니다. answer 분포, validation loss, 실제 제출 점수를 같이 봤습니다.
5. Qwen3-VL-8B가 초기 실험에서 가장 점수가 높았기 때문에 Qwen3-VL-32B 확장이 다음 실험으로 이어졌습니다.
