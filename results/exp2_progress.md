# Exp2 진행 스냅샷

**범위:** seed 42의 사전 고정 held-out 80개 가운데 13개 사례(52 arm rows)가 모두 완료됐습니다. LoCoMo 10개와 LongMemEval 3개이며, 아직 전체 benchmark가 아닌 중간 진행상황입니다. 원시 JSONL은 모든 80개가 끝난 뒤 공개할 예정입니다.

| Arm | Judge correct | Accuracy | Mean estimated USD/case | Mean logical LLM calls |
|---|---:|---:|---:|---:|
| joint | 6/13 | 0.462 | 0.0644 | 667.7 |
| sequential | 6/13 | 0.462 | 0.0874 | 845.5 |
| delete_only | 6/13 | 0.462 | 0.0592 | 667.7 |
| add_only | 6/13 | 0.462 | 0.0241 | 311.5 |

13개 사례 모두 네 arm이 완료됐습니다. 10개 LoCoMo 사례는 두 대화 그룹뿐이고 나머지 3개는 LongMemEval이므로, 표본은 작고 탐색적입니다. 지표는 GPT judge 기준입니다. 정책 우열/통계적 유의성 결론은 내리지 않습니다. 비용은 코드 설정 단가를 이용한 추정입니다.

추가 텍스트 지표(현재 13개 prediction, 각 arm):

| Arm | EM | Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-3 |
|---|---:|---:|---:|---:|---:|
| joint | 0.077 | 0.352 | 0.352 | 0.121 | 0.073 |
| sequential | 0.077 | 0.352 | 0.352 | 0.121 | 0.073 |
| delete_only | 0.077 | 0.346 | 0.346 | 0.102 | 0.053 |
| add_only | 0.077 | 0.363 | 0.363 | 0.121 | 0.073 |

기존 저장 prediction을 API 호출 없이 새 EM/F1/ROUGE 코드로 다시 점수화하면서 EM도 갱신했습니다. Judge와 exact string 지표의 차이는 정의 차이입니다. BERTScore는 optional `bert-score` 패키지가 없어 아직 산출하지 않았습니다.

입력 상태: `results/manifest.json`의 80개 테스트 ID, `results/exp2_test.jsonl`의 사례별 JSONL 체크포인트. 다음 실행은 동일 output에 이어서 진행합니다.
