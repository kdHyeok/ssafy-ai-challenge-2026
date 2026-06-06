# Data

## 현재 포함 파일

| 경로 | 내용 | 역할 |
|---|---|---|
| `train.csv` | `id, path, question, a, b, c, d, answer`를 열로 가진 5,073행의 데이터 | 학습 입력 형식의 데이터 |
| `dev.csv` | `id, path, question, a, b, c, d, answer1~answer5`를 열로 가진 4,413행의 데이터 | 다수 응답 구조의 추가 데이터 |
| `test.csv` | `id, path, question, a, b, c, d`를 열로 가진 5,074행의 데이터 | 추론 입력 형식의 데이터 |
| `example_submission.csv` | `id, answer`를 열로 가진 5,074행의 데이터 | 제출 파일 예시(실제는 비어있는 answer 열에 a,b,c,d를 추론) |
| `train/` | train 이미지 샘플 10장 | 학습 이미지 경로 예시 |
| `dev/` | dev 이미지 샘플 10장 | 개발 이미지 경로 예시 |
| `test/` | test 이미지 샘플 10장 | 검증 이미지 경로 예시 |

- [Dataset Description](../docs/Dataset%20Description.md)

이 레포지토리의 데이터는 이미지 일부만 포함한 샘플 데이터로, 실제 대회의 이미지 데이터는 용량 문제로 일부 생략하였습니다.
