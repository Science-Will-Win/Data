# 03_PRM_채점

PRM(Process Reward Model, 과정 보상 모델) 채점 결과입니다.

## 개요

PRM은 에이전트의 풀이 과정을 단계별로 평가합니다.
비유: "수학 시험에서 최종 답뿐 아니라 풀이 과정 각 줄에 점수를 매기는 것"

## 채점 방법: 이중 판정관 (Dual-Judge)

| 판정관 | 모델 | 역할 |
|--------|------|------|
| 판정관 1 | Claude Sonnet 4 | 외부 평가 |
| 판정관 2 | GPT-4.1 | 자가 평가 |

## 핵심 분석 결과

| 지표 | 설명 | 판별력 |
|------|------|--------|
| `last3_avg` | 마지막 3단계 평귰 점수 | ⭐⭐⭐ 높음 |
| `low_ratio` | 0.3 미만 단계 비율 | ⭐⭐⭐ 높음 |
| `prm_avg` | 전체 평기 점수 | ⭐⭐ 보통 |
| `prm_min` | 최저 점수 | ⭐ 낮음 |

## Uncertainty Head 개선 방향 (완수 참고)

현재 Surrogate Head: 정확도 63.2%, 불확실 탐지율 81%

**개선 방향:**
1. PRM `last3_avg`, `low_ratio`를 특징으로 추가
2. 모델 은닉 상태(Hidden State) 기반 Neural Head
3. Surrogate + Neural + PRM 앙상블
