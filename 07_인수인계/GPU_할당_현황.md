# GPU 할당 현황

## GPU 6-7번만 사용 가능

| GPU | 할당 |
|-----|------|
| GPU 0-5 | ❌ 다른 연구자 (절대 사용 금지) |
| **GPU 6** | ✅ AIGEN BioAgent (A100 80GB) |
| **GPU 7** | ✅ AIGEN BioAgent (A100 80GB) |

## 규칙
1. GPU 0-5 프로세스 절대 중지/변경 금지
2. 항상 `CUDA_VISIBLE_DEVICES=6,7`로 제한
3. vLLM: `--tensor-parallel-size 2`로 GPU 2개 사용

## 확인
```bash
nvidia-smi -i 6,7
```
