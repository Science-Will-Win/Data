# AIGEN BioAgent Phase 1 — Trajectory 데이터 전달 문서

**전달일:** 2026년 2월 11일  
**작성:** Hoony (Phase 1 개발 리드)  
**수신:** 박재현 (Phase 2 개발)  

---

## 1. 개요

Phase 1(Aigen-E1)에서 Biomni-Eval1 벤치마크 433개 태스크를 대상으로 수집한 Trajectory 데이터를 Phase 2(Aigen-R0) 학습용으로 전달한다. 에이전트는 Biomni-R0-32B-AWQ-INT4 모델을 사용하여 vLLM으로 추론하며, 각 태스크에 대해 think → execute → observe → solution 흐름의 Trajectory를 생성했다.

---

## 2. 수집 결과 요약

### 2.1 전체 통계

| 항목 | 건수 | 비율 |
|------|------|------|
| 벤치마크 총 태스크 | 433개 | 100% |
| 실행 완료 | 277건 | 64% |
| **성공 (score=1.0)** | **96건** | **35%** |
| 실패 (score=0.0) | 163건 | 59% |
| 에러 (score=-1.0) | 18건 | 6% |

### 2.2 태스크 유형별 성공률

| 태스크 유형 | 전체 | 성공 | 성공률 |
|------------|------|------|--------|
| gwas_variant_prioritization (변이 우선순위) | 52 | 31 | 60% |
| lab_bench_seqqa (서열 QA) | 50 | 30 | 60% |
| screen_gene_retrieval (스크리닝 유전자) | 50 | 13 | 26% |
| gwas_causal_gene_pharmaprojects | 50 | 12 | 24% |
| patient_gene_detection (환자 유전자 탐지) | 50 | 8 | 16% |
| rare_disease_diagnosis (희귀질환 진단) | 25 | 2 | 8% |
| **합계** | **277** | **96** | **35%** |

---

## 3. 전달 파일

| 파일명 | 건수 | 용량 | 용도 |
|--------|------|------|------|
| `phase2_delivery_all.jsonl` | 277건 | 14MB | 전체 원본 (성공+실패+에러) |
| `phase2_delivery_positive_fixed.jsonl` | 96건 | 3.5MB | **SFT 학습용** (성공 케이스만) |
| `phase2_delivery_failure_fixed.jsonl` | 163건 | 9.8MB | GDPO 부정 예제 (score=0.0) |
| `phase2_delivery_error_fixed.jsonl` | 18건 | 87KB | 에러 케이스 (score=-1.0) |

---

## 4. 데이터 포맷 상세

### 4.1 JSONL 필드 구조

| 필드 | 타입 | 설명 |
|------|------|------|
| `task_id` | string | 태스크 고유 ID (예: `eval1_gwas_variant_prioritization_134`) |
| `benchmark` | string | 벤치마크 이름 (`biomni-eval1`) |
| `task_type` | string | 태스크 유형 (6개 카테고리) |
| `question` | string | 원본 프롬프트 (벤치마크 문제) |
| `ground_truth` | string | 정답 (평가 기준) |
| `trajectory` | array | `[{step, content}]` — 에이전트의 전체 실행 과정 |
| `final_answer` | string | 에이전트가 추출한 최종 답변 |
| `score` | float | 1.0(성공), 0.0(실패), -1.0(에러) |
| `success` | boolean | true/false |
| `metadata` | object | execution_time_sec, num_steps, timestamp |

### 4.2 Trajectory 구조

1. **Step 1:** Human Message — 벤치마크 문제
2. **Step 2:** AI Message — `<think>추론</think>` + `<execute>코드실행</execute>`
3. **Step 3:** Observation — 도구 실행 결과
4. **Step 4~N:** AI → Observation 반복 (다단계 추론)
5. **Final Step:** AI Message — `<solution>최종답변</solution>`

---

## 5. Phase 2 학습 활용 가이드

### 5.1 SFT 학습 데이터 변환
```python
import json

positive = [json.loads(l) for l in open("phase2_delivery_positive_fixed.jsonl")]
sft_data = [
    {"instruction": d["question"], "output": d["trajectory"][-1]["content"]}
    for d in positive
]
```

### 5.2 GDPO 학습 활용

- `positive_fixed.jsonl` (96건) → 긍정 예제 (chosen)
- `failure_fixed.jsonl` (163건) → 부정 예제 (rejected)
- 같은 `task_type` 내에서 성공/실패 쌍 매칭

### 5.3 리즈닝 토큰 매핑
```python
content = content.replace("<think>", "[THINK]").replace("</think>", "[/THINK]")
```

---

## 6. 서버 경로

| 항목 | 경로 |
|------|------|
| 전체 데이터 | `/home/sww/aigen-bioagent/trajectory_output/phase2_delivery_all.jsonl` |
| SFT용 | `/home/sww/aigen-bioagent/trajectory_output/phase2_delivery_positive_fixed.jsonl` |
| GDPO용 | `/home/sww/aigen-bioagent/trajectory_output/phase2_delivery_failure_fixed.jsonl` |

**서버 접속:** `ssh -p 22 sww@211.206.241.242`

---

## 7. 향후 계획

1. GPU 7: LAB-Bench (1,554개) — 수집 완료 후 추가 전달
2. BioML-Bench (24개) — 별도 파이프라인 필요
3. 성공률 향상 후 재수집 예정

포맷 불일치 시 Hoony에게 연락 바랍니다.

---

*AIGEN Sciences & Modulabs 내부용*
