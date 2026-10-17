# exp2_200_run 중단 진단

- 200개 사례를 요청한 manifest에 비해 결과 JSONL에는 10개 사례(40 arm rows)만 완료되어 있었고, 모두 LoCoMo의 첫 두 대화 그룹에 속했습니다.
- 완료한 결과에서 구조화 프롬프트/인덱스 오류 징후는 없었습니다. 모든 arm이 기록되었고 `invalid_indices`는 0입니다.
- 가장 가능성 높은 병목은 계산량입니다. 각 독립 QA가 전체 대화 history를 cold-start로 처리해 사례당 arm별 수백~천 회 모델 호출이 발생했습니다. 사례가 끝날 때만 JSONL을 쓰므로 처리 중 정지처럼 보일 수 있습니다.
- 로그가 없어 수동 종료, 런타임 제한, API quota/network 중 무엇이 직접 원인인지는 확정할 수 없습니다.
- resume는 완료한 arm 결과를 보존하도록 고쳤고, 데이터셋 교차 실행 `--exp2-order interleave`를 추가했습니다. API 재시도 실패에는 label/cache key가 기록됩니다.
- 단계 실행용 `--exp2-start-index`도 추가했습니다. `--exp2-n`은 manifest에서 총 선택할 held-out 수(기본 80)로 두고, start index로 실행 구간을 정합니다.

테스트: `python3 -m unittest -v test_smoke.py` (9개 통과).

실제 1-case API smoke 시도는 실행 제한을 두 번 초과했고 JSONL score row를 남기지 못했습니다. 응답 캐시 124개는 `results/exp2_live_smoke/`에 남아 재개 가능하지만 공개하지 않았습니다. 기존 부분 JSONL을 결과로 혼합하지 않았습니다.
