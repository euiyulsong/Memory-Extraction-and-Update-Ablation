# Exp2 진행 스냅샷

**범위:** seed 42의 사전 고정 held-out 80개 가운데 14개 사례(56 arm rows)가 모두 완료됐습니다. LoCoMo 10개와 LongMemEval 4개이며, 아직 전체 benchmark가 아닌 중간 진행상황입니다. 원시 JSONL에는 14개 완료 사례만 반영했습니다.

| Arm | Judge correct | Accuracy | Mean estimated USD/case | Mean logical LLM calls |
|---|---:|---:|---:|---:|
| joint | 7/14 | 0.500 | 0.0644 | 667.7 |
| sequential | 7/14 | 0.500 | 0.0874 | 845.5 |
| delete_only | 7/14 | 0.500 | 0.0592 | 667.7 |
| add_only | 7/14 | 0.500 | 0.0241 | 311.5 |

14개 사례 모두 네 arm이 완료됐습니다. 10개 LoCoMo 사례는 두 대화 그룹뿐이고 나머지 4개는 LongMemEval이므로, 표본은 작고 탐색적입니다. 지표는 GPT judge 기준입니다. 정책 우열/통계적 유의성 결론은 내리지 않습니다. 비용은 코드 설정 단가를 이용한 추정입니다.

추가 텍스트 지표(현재 14개 prediction, 각 arm):

| Arm | EM | Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-3 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint | 0.143 | 0.398 | 0.398 | 0.112 | 0.068 | 0.374 | 0.381 |
| sequential | 0.143 | 0.398 | 0.398 | 0.112 | 0.068 | 0.374 | 0.381 |
| delete_only | 0.143 | 0.392 | 0.392 | 0.094 | 0.049 | 0.369 | 0.396 |
| add_only | 0.143 | 0.409 | 0.409 | 0.112 | 0.068 | 0.385 | 0.408 |

기존 저장 prediction을 API 호출 없이 재점수화했습니다. BERTScore는 `roberta-large`, English, baseline-rescaled F1 설정입니다. Judge와 exact string 지표의 차이는 정의 차이입니다.

입력 상태: `results/manifest.json`의 80개 테스트 ID, `results/exp2_test.jsonl`의 사례별 JSONL 체크포인트. 다음 실행은 동일 output에 이어서 진행합니다.
