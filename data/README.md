# Data

이 폴더는 대회 데이터 구조와 공개 노트북의 입출력 형식을 설명하기 위한 공개 샘플만 포함합니다. 원본 전체 CSV, 전체 이미지, 학습된 모델 weight/checkpoint는 포함하지 않습니다.

## 현재 포함 파일

| 경로 | 내용 | 공개 레포에서의 역할 |
|---|---|---|
| `train.csv` | `id, path, question, a, b, c, d, answer` 형식의 10행 샘플 | 학습 데이터 로드 구조 설명 |
| `dev.csv` | `id, path, question, a, b, c, d, answer1~answer5` 형식의 10행 샘플 | dev 응답 불일치와 agreement 분석 구조 설명 |
| `test.csv` | `id, path, question, a, b, c, d` 형식의 10행 샘플 | 제출 대상 입력 구조 설명 |
| `sample_submission_format.csv` | `id, answer` 형식의 10행 샘플 | 대회 제출 CSV 포맷 예시 |
| `train/` | `train_0001.jpg` ~ `train_0010.jpg` | 학습 이미지 경로 구조를 보여주는 샘플 10장 |
| `dev/` | `dev_0001.jpg` ~ `dev_0010.jpg` | dev 이미지 경로 구조를 보여주는 샘플 10장 |
| `test/` | `test_0001.jpg` ~ `test_0010.jpg` | test 이미지 경로 구조를 보여주는 샘플 10장 |

## CSV 구조

- `train.csv`: 정답이 하나인 supervised 학습 데이터 구조입니다.
- `dev.csv`: 다수 응답(`answer1`~`answer5`)이 있어, 다수결·응답 일치도·pseudo-label 후보를 분석하는 데 사용했습니다.
- `test.csv`: 정답 없이 질문과 보기만 제공되며, 모델 예측을 `sample_submission_format.csv` 형식으로 제출합니다.

## 공개 범위와 주의점

- 이 폴더의 CSV와 이미지는 모두 split별 10개 샘플로 맞춰 둔 공개용 예시입니다.
- 공개 노트북은 데이터 로드 방식과 실험 흐름을 설명하기 위한 대표 자료입니다. 전체 학습/추론을 재현하려면 대회에서 제공한 원본 데이터 전체와 실행 환경이 필요합니다.
- 모델 weight, checkpoint, 중간 확률 CSV, 실제 최고점권 제출 CSV 전체는 용량과 공개 범위 문제로 포함하지 않습니다.

## 데이터 활용 방식

- `dev.csv`는 다중 응답 구조를 가지므로, 응답 일치도와 질문 유형을 함께 확인해 pseudo-label 후보와 실험 해석에 활용했습니다.
- `test.csv`는 정답이 없는 평가 입력이므로, 후처리와 앙상블 비교에서는 예측 confidence와 제출 간 answer disagreement를 함께 참고했습니다.
- 이 폴더는 구조 이해용 샘플이며, 전체 데이터 공개를 의미하지 않습니다.
