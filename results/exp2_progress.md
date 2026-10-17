# Exp2 진행 스냅샷

**범위:** seed 42의 사전 고정 held-out 80개 가운데 12개 사례(48 arm rows)가 모두 완료됐습니다. LoCoMo 10개와 LongMemEval 2개이며, 아직 전체 benchmark가 아닌 중간 진행상황입니다. 원시 JSONL은 모든 80개가 끝난 뒤 공개할 예정입니다.

| Arm | Judge correct | Accuracy | Mean estimated USD/case | Mean logical LLM calls |
|---|---:|---:|---:|---:|
| joint | 5/12 | 0.417 | 0.0690 | 717.0 |
| sequential | 5/12 | 0.417 | 0.0938 | 908.3 |
| delete_only | 5/12 | 0.417 | 0.0635 | 717.0 |
| add_only | 5/12 | 0.417 | 0.0257 | 334.3 |

12개 사례 모두 네 arm이 완료됐습니다. 10개 LoCoMo 사례는 두 대화 그룹뿐이고 나머지 2개는 LongMemEval이므로, 표본은 작고 탐색적입니다. 지표는 GPT judge 기준입니다. 정책 우열/통계적 유의성 결론은 내리지 않습니다. 비용은 코드 설정 단가를 이용한 추정입니다.

입력 상태: `results/manifest.json`의 80개 테스트 ID, `results/exp2_test.jsonl`의 사례별 JSONL 체크포인트. 다음 실행은 동일 output에 이어서 진행합니다.
