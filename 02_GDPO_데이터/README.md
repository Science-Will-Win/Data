# 02_GDPO_데이터

Phase 2 학습에 직접 사용되는 핵심 데이터셋입니다.

## 파일 목록

| 파일 | 건수 | 설명 |
|------|------|------|
| `gdpo_unified.jsonl` | 829쌍 | GDPO 선호도 학습용 chosen/rejected 쌍 |
| `sft_unified.jsonl` | 444건 | SFT(지도 미세조정) 학습용 정답 데이터 |
| `uncertainty_labels.json` | 433건 | 태스크별 난이도 + 거부 권장 라벨 |

## 데이터 스키마

### gdpo_unified.jsonl
```json
{
  "task_id": "gwas_variant_prioritization_134",
  "task_type": "gwas_variant_prioritization",
  "benchmark": "eval1",
  "chosen": "정답에 도달한 궤적 (좋은 풀이 과정)",
  "rejected": "오답이거나 실패한 궤적 (나쁜 풀이 과정)",
  "ground_truth": "rs4253311",
  "scenario": "B",
  "source": "gpt_teaches_r0"
}
```

**시나리오 분류:**
| 시나리오 | 의미 | 건수 |
|---------|------|------|
| A | R0 정답, GPT 오답 → R0가 가르침 | 23 |
| B | GPT 정답, R0 오답 → GPT가 가르침 | 664 |
| C | 둘 다 정답, R0가 더 나음 | 47 |

### sft_unified.jsonl
```json
{
  "task_id": "gwas_variant_prioritization_134",
  "task_type": "gwas_variant_prioritization",
  "benchmark": "eval1",
  "answer": "rs4253311",
  "ground_truth": "rs4253311",
  "model": "gpt",
  "score": 1
}
```

### uncertainty_labels.json
```json
{
  "task_id": "gwas_variant_prioritization_134",
  "difficulty": "hard",
  "should_refuse": true,
  "reason": "GWAS 태스크는 높은 판단 실패율 보유"
}
```

## 데이터 생성 과정
```
Eval1 (433건) + LAB-Bench (1,967건)
    ↓
R0 수집 + GPT-4.1 수집
    ↓
정답 비교 → A/B/C/D 시나리오 분류
    ↓
Eval1 287쌍 + LAB-Bench 542쌍 = 829쌍
```
