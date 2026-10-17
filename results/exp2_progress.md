# Exp2 진행 스냅샷

**범위:** seed 42의 held-out 80개 가운데 현재 30개 사례(120 arm rows)가 네 arm 모두 완료됐습니다. LoCoMo 15개와 LongMemEval 15개입니다. 나머지 50개는 아직 미완료입니다.

별도 exploratory smoke run은 정식 held-out과 분리하고, 같은 QA가 일부 겹칠 수 있으므로 본 통계에 포함하지 않습니다.

| Arm | Judge correct | Accuracy | Mean estimated USD/case | Mean logical LLM calls |
|---|---:|---:|---:|---:|
| joint | 18/35 | 0.514 | 0.0582 | 631.9 |
| sequential | 17/35 | 0.486 | 0.0792 | 799.1 |
| delete_only | 17/35 | 0.486 | 0.0544 | 633.9 |
| add_only | 17/35 | 0.486 | 0.0226 | 303.7 |

30개 사례 모두 원본 `results/manifest.json`의 held-out 목록에 포함되며, 완료된 benchmark 사례 수로 집계했습니다. Judge는 모델 평가이며, 이 중간값만으로 정책 우열을 단정하지 않습니다. 비용은 코드 단가 기반 추정입니다.

추가 텍스트 지표(완료 30개 prediction, 각 arm):

| Arm | EM | Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-3 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 0.257 | 0.427 | 0.427 | 0.192 | 0.113 | 0.417 | 0.445 |
| sequential | 0.257 | 0.400 | 0.400 | 0.202 | 0.113 | 0.391 | 0.441 |
| delete_only | 0.257 | 0.395 | 0.395 | 0.195 | 0.105 | 0.386 | 0.440 |
| add_only | 0.257 | 0.402 | 0.402 | 0.202 | 0.113 | 0.392 | 0.446 |

기존 저장 prediction을 API 호출 없이 재점수화했습니다. BERTScore는 `roberta-large`, English, baseline-rescaled F1 설정입니다. Judge와 exact string 지표의 차이는 정의 차이입니다.

입력 상태: `results/manifest.json`의 80개 테스트 ID, `results/exp2_test.jsonl`의 사례별 JSONL 체크포인트. 다음 실행은 동일 output에 이어서 진행합니다.
