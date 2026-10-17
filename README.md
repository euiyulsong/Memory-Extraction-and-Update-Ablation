# Long-Term Memory Extraction & Update Ablation

LoCoMo와 LongMemEval 일부를 이용해 장기 메모리의 추출 단위와 저장/삭제 정책을 비교하는 재현 가능한 벤치마크입니다. 원 실험 코드는 [`memory_eval_benchmark`](https://github.com/euiyulsong)에서 시작했으며, **2026-10-15 현재 완료된 개발셋 결과까지만** 이 저장소에 정리했습니다.

> **진행 상태:** Exp 1 개발셋 20개 QA × 2 조건(총 40행)은 완료. Exp 2 held-out 80개 QA는 시작 전이며 결과 파일이 비어 있습니다. 따라서 아래 결과는 최종 벤치마크 결론이 아니라 방법 선택용 중간 결과입니다.

## 실험 설계

고정 seed 42로 총 100개 QA를 선택합니다(LoCoMo 50, LongMemEval 50). 개발셋 20개(각 데이터셋 10개)에서 추출 단위를 비교하고, 선택한 조건으로 나머지 80개를 평가하도록 설계했습니다. LoCoMo는 대화 단위로 분할합니다.

| 조건 | 추출 단위 | 저장 정책 | 검색 |
|---|---|---|---|
| `turn_1` | 대화 turn 1개씩 | add-only | BM25 top-8 |
| `turn_5` | 연속 turn 최대 5개(세션 경계 유지) | add-only | BM25 top-8 |

두 조건은 동일한 모델 및 JSON 추출 스키마를 사용합니다. Exp 1 선택 우선순위는 개발셋 `correct_rate` → token F1 → 호출 수입니다. 후속 Exp 2는 선택된 추출 단위를 고정하고 `joint`, `sequential`, `delete_only`, `add_only` 업데이트 정책을 80개 held-out QA에서 비교하도록 구현되어 있습니다.

### 데이터

- [LoCoMo](https://github.com/snap-research/locomo): 원본 대화 전체, 10개 대화에서 QA를 선택.
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval): `longmemeval_oracle.json`; 관련 evidence session만 제공하는 oracle 버전.
- 샘플 ID와 분할: [`results/manifest.json`](results/manifest.json)

LongMemEval oracle은 전체 noisy history가 아니라 근거 세션을 미리 제공하므로 검색이 실제 서비스보다 쉬울 수 있습니다. 결과는 영어 중심 공개 데이터셋에서 얻은 탐색적 결과이며 한국어/production 일반화로 해석하면 안 됩니다.

## 현재까지의 결과: Exp 1 개발셋

실행 산출물: [`results/exp1_dev.jsonl`](results/exp1_dev.jsonl) (20 QA × 2 조건). 아래 수치는 이 20개 개발 QA의 요약이며 held-out 테스트 결과가 아닙니다.

| 데이터셋 | 방식 | 정답 (GPT judge) | 평균 EM | 평균 Subspan EM | 평균 token F1 | Evidence-origin recall@8 | 평균 활성 메모리 수 | 총 모델 호출 | 추정 비용 (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 | `turn_1` | 9/20 (45%) | 0.150 | 0.300 | 0.430 | 0.708 | 668.3 | 7,023 | $0.5280 |
| 전체 | `turn_5` | 7/20 (35%) | 0.150 | 0.250 | 0.380 | 0.700 | 507.3 | 1,576 | $0.2560 |
| LoCoMo | `turn_1` | 2/10 (20%) | 0.000 | 0.100 | 0.244 | 0.650 | 653.5 | 6,735 | $0.4966 |
| LoCoMo | `turn_5` | 1/10 (10%) | 0.000 | 0.000 | 0.175 | 0.550 | 503.5 | 1,490 | $0.2369 |
| LongMemEval oracle | `turn_1` | 7/10 (70%) | 0.300 | 0.500 | 0.616 | 0.900 | 28.2 | 288 | $0.0314 |
| LongMemEval oracle | `turn_5` | 6/10 (60%) | 0.300 | 0.500 | 0.586 | 0.850 | 21.0 | 86 | $0.0191 |

**중간 관찰:** 사전 정의한 1차 지표인 개발셋 정확도에서 `turn_1`이 `turn_5`보다 2/20 QA 높았습니다. Exact Match는 두 방식 모두 3/20(15%)이며, Subspan EM은 각각 6/20(30%)과 5/20(25%)입니다. Subspan EM은 정규화된 gold 토큰 전체가 prediction 안에 연속된 구간으로 나타나는지를 측정합니다. Evidence-origin recall@8은 전체 QA 평균으로 `turn_1` 0.708, `turn_5` 0.700입니다. 반면 `turn_5`는 모델 호출이 약 4.5배 적고 추정 비용이 약 51% 낮았습니다. 작은 표본과 동일 모델 기반 judge를 고려하면 확정적 우열로 볼 수 없으며, Exp 2는 계획된 프로토콜상 `turn_1`을 사용해야 합니다.

비용은 코드의 기본 단가(input $0.10/1M, cached input $0.01/1M, output $0.50/1M)를 사용한 **논리적 cold-start 추정치**입니다. 실제 청구액이나 production 반복 질의 비용과 같지 않습니다. 정답은 GPT 기반 judge 판정이며 독립 인간 평가가 아닙니다. Evidence-origin recall은 출처 turn 회수율이지 사실 정합성 점수가 아닙니다.

## 재현

Python 3.10 이상 및 OpenAI API key가 필요합니다.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='YOUR_API_KEY'
python memory_benchmark.py --model gpt-6-luna \
  --seed 42 --topk 8 --update-topk 12 \
  --data-dir data --output-dir results
```

데이터는 필요 시 자동 다운로드됩니다. API 응답 캐시는 `results/api_cache/`에 남아 이어서 실행할 수 있습니다. 공개한 부분 결과를 원본 입력과 대조하려면 해당 저장소의 `manifest.json` 샘플 ID가 기준입니다. API 키 없이 합성 데이터 스모크 테스트를 하려면 원 프로젝트의 `test_smoke.py`를 함께 사용하세요.

### 산출물

- `results/manifest.json`: 샘플/분할 manifest
- `results/exp1_dev.jsonl`: 완료된 개발셋 결과
- `results/exp2_test.jsonl`: held-out 평가 결과 (현재 아직 미실행)
- `memory_benchmark.py`: 실험 및 후속 단계 실행 코드

원본 실행 디렉터리의 API cache 및 전체 history audit 로그는 크기와 불필요한 중간 세부정보 때문에 이 저장소에 포함하지 않았습니다.

## Exp 2 진행 결과

held-out 80개 중 현재 **완료 10개 사례**(LoCoMo 10/10, 각 arm 결과 10개; LongMemEval은 아직 없음)입니다. 11번째 사례에는 `joint`, `sequential`만 기록됐고 나머지 arm은 미완료입니다. 공정한 paired 비교에서 이 사례를 제외하면 네 arm 모두 동일 정답 5/10 (50%)입니다. 이 작은 진행분은 LoCoMo 대화 두 개에만 걸쳐 있어 정책 우열로 해석하면 안 됩니다. 진행 요약은 [`results/exp2_progress.md`](results/exp2_progress.md)에 있습니다. 전체 JSONL은 80개가 완료된 뒤 추가합니다.

| Arm | Correct / n | 평균 사례당 추정비용 (USD) | 평균 논리 호출 수 |
|---|---:|---:|---:|
| joint | 5/10 paired | 0.0817 | 850.0 |
| sequential | 5/10 | 0.1112 | 1,077.0 |
| delete_only | 5/10 | 0.0751 | 850.0 |
| add_only | 5/10 | 0.0303 | 396.0 |

평균 비용은 코드의 모델 단가 추정값이며 실제 청구액과 다를 수 있습니다. 기존 Exp2 결과 JSONL과 탐색용 200개 파일은 코드/캐시와 구분해 아직 Git에 포함하지 않았습니다.

## exp2 중단 진단 및 수정

확장 실험 manifest는 200개 사례를 요청했지만 JSONL에는 10개 사례(40 arm rows)만 완료되어 있었고 모두 LoCoMo였습니다. 대화 이력을 QA마다 cold-start로 다시 처리하기 때문에 사례당 arm별 수백~천 회의 모델 호출이 필요할 수 있습니다. 로그가 없어 수동 종료/런타임 종료/API 실패 등 직접 원인은 단정할 수 없습니다. 완료된 행은 모두 네 arm을 갖고 `invalid_indices=0`이라, 현재 증거는 프롬프트 오류보다 실행 시간/규모 문제를 더 시사합니다.

resume 시 완료 arm을 유지하고 미완료 arm만 돌도록 수정했습니다. `--exp2-order interleave`로 두 데이터셋 결과가 장시간 실행 초기에 모두 생기게 했습니다. 상세 기록은 [`results/exp2_200_run/diagnosis.md`](results/exp2_200_run/diagnosis.md) 참조. 실 API 소규모 시도는 실행 제한에 걸려 score row를 만들지 못했으며 결과로 취급하지 않았습니다. API 응답 캐시는 `results/exp2_live_smoke/`에 남아 있어 아래 명령으로 재개할 수 있습니다. 캐시/부분 산출물은 ignore되어 저장소에 포함되지 않습니다.

```bash
python3 memory_benchmark.py --model gpt-6-luna --exp2-only --exp2-n 1 \\
  --exp2-order interleave --exp2-outdir results/exp2_live_smoke \\
  --fresh-exp2-output results/exp2_live_smoke/exp2_smoke.jsonl
```

## 다음 단계

## 다음 단계

1. 동일 seed/조건으로 Exp 2의 held-out 80개 QA에서 네 업데이트 정책을 실행. 실행 제한에 맞춰 `--exp2-n 1 --exp2-start-index 0..79`로 나눠서 진행할 수 있으며, 같은 output JSONL을 사용해 재개합니다.
2. `summary.csv`, `report.json`의 paired cluster-bootstrap CI를 검토하고 결과를 추가.
3. 대표 오류를 수동 검토하고, LongMemEval full-history 및 한국어 데이터에서 검증.

## 출처와 제한

- [LoCoMo](https://github.com/snap-research/locomo)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval), ICLR 2025
- 모델은 실행 설정에 따라 지정되며 기본값은 `gpt-6-luna`입니다.
- 데이터셋 라이선스/이용 조건과 해당 API 제공자의 정책을 준수하세요.
