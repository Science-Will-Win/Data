# 05_모델_체크포인트

모델 파일은 대용량이므로 서버에만 존재합니다.

## 모델 목록

| 모델 | 크기 | 서버 경로 |
|------|------|-----------|
| Biomni-R0-32B-AWQ | ~18GB | `/home/sww/models/Biomni-R0-32B-AWQ-INT4/` |
| GDPO 병합 모델 | 62GB | `/home/sww/models/Qwen3-32B-GDPO-merged/` |
| GDPO LoRA 어댑터 | 4.3GB | `gdpo_output/final/` |
| Uncertainty Head | 12KB | `uncertainty_head_output/` |
| Ministral 3B | 8.8GB | `/home/sww/models/Ministral-3-3B-Instruct/` |

## GDPO 학습 설정

| 항목 | 값 |
|------|-----|
| 베이스 모델 | Qwen3-32B |
| 양자화 | QLoRA 4-bit NF4 |
| LoRA r/alpha | 64/64 |
| 학습률 | 5e-5 |
| 에포크 | 3 |
| 총 스텝 | 273 |
| 평가 정확도 | 88.9% |
| 학습 시간 | 약 2시간 50분 |

## vLLM 서버 실행

```bash
# Base R0 (주의: --quantization awq 쓰면 에러!)
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
  --model /home/sww/models/Biomni-R0-32B-AWQ-INT4 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --port 8003

# GDPO 병합 모델
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
  --model /home/sww/models/Qwen3-32B-GDPO-merged \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --port 8002
```
