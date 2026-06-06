# Data

이 폴더는 대회 데이터 구조와 공개 노트북의 입출력 형식을 설명하기 위한 샘플만 포함합니다. 원본 전체 CSV, 전체 이미지, 학습된 모델 weight/checkpoint는 포함하지 않습니다.

## 현재 포함 파일

| 경로 | 내용 | 역할 |
|---|---|---|
| `train.csv` | `id, path, question, a, b, c, d, answer` 구조의 train 샘플 10행 | 학습 입력 형식 예시 |
| `dev.csv` | `id, path, question, a, b, c, d, answer1~answer5` 구조의 dev 샘플 10행 | dev 다수 응답 구조 예시 |
| `test.csv` | `id, path, question, a, b, c, d` 구조의 test 샘플 10행 | 추론 입력 형식 예시 |
| `sample_submission_format.csv` | `id, answer` 구조의 제출 형식 10행 | 제출 CSV 포맷 예시 |
| `train/` | train 이미지 샘플 10장 | 이미지 경로 확인용 |
| `dev/` | dev 이미지 샘플 10장 | 이미지 경로 확인용 |
| `test/` | test 이미지 샘플 10장 | 이미지 경로 확인용 |

## 이미지 경로

CSV의 `path` 컬럼은 이 폴더 기준 상대 경로입니다.

예시:

```text
train/train_0001.jpg
dev/dev_0001.jpg
test/test_0001.jpg
```

## 주의점

- 이 폴더의 CSV는 구조 설명을 위한 10행 샘플입니다.
- split별 이미지도 각 10장만 포함합니다.
- 모델 weight, checkpoint, 중간 확률 CSV, 실제 제출 CSV 전체는 포함하지 않습니다.
- 점수 재검토와 실험 해설은 `README.md`와 `docs/` 문서를 기준으로 봐야 합니다.
- 공개 노트북을 그대로 실행하려면 원본 대회 데이터와 Colab/GPU 환경이 별도로 필요합니다.
