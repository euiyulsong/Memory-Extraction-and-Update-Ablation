# Exp2 진행 스냅샷

**범위:** seed 42의 사전 고정 held-out 80개 가운데 첫 10개 완료 사례. 이들은 모두 LoCoMo이며 두 대화 그룹(conv-26, conv-30)에 속합니다. 따라서 아래 숫자는 중간 진행상황이지 최종 benchmark 비교가 아닙니다. 결과 JSONL은 아직 GitHub에 공개하지 않았고, 전체 80개 완료 후 함께 올릴 예정입니다.

| Arm | Judge correct | Accuracy | Mean estimated USD/case | Mean logical LLM calls |
|---|---:|---:|---:|---:|
| joint | 5/10 complete paired cases | 0.50 | 0.0817 | 850.0 |
| sequential | 5/10 | 0.50 | 0.1112 | 1,077.0 |
| delete_only | 5/10 | 0.50 | 0.0751 | 850.0 |
| add_only | 5/10 | 0.50 | 0.0303 | 396.0 |

JSONL에 11번째 사례의 `joint` row만 추가로 기록되어 있으나, arm 3개가 없어 모든 arm 비교에서 제외했습니다. 완결 사례 10개에서는 각 arm이 5/10이었습니다. 지표는 GPT judge 기준이고, LoCoMo 대화 클러스터 2개뿐이라 정책 우열/통계적 유의성 결론을 낼 수 없습니다. 비용은 코드 설정 단가를 이용한 추정입니다.

입력 상태: `results/manifest.json`의 80개 테스트 ID, `results/exp2_test.jsonl`의 사례별 JSONL 체크포인트. 다음 실행은 동일 output에 이어서 진행합니다.
