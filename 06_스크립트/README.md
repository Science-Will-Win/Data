# 06_스크립트

프로젝트에서 사용한 핵심 스크립트입니다.

## 파일 목록

| 스크립트 | 용도 | GPU | 시간 |
|---------|------|-----|------|
| `train_uncertainty_head.py` | 불확실성 헤드 MLP | CPU 가능 | ~1분 |
| `train_ministral_sft.py` | Ministral 3B SFT | A100×1 | ~30분 |
| `run_phase_e.py` | Phase E 비교 실험 | vLLM 필요 | ~50분/조건 |
| `analyze_phase_e.py` | 결과 분석 | CPU만 | ~1분 |

## 실행 환경

```bash
ssh sww@211.206.241.242 -p 22
conda activate biomni-agent
cd ~/aigen-bioagent
```

## 실행 예시

```bash
# Phase E 실험
python run_phase_e.py --condition base_r0
python run_phase_e.py --condition gdpo
python run_phase_e.py --condition gdpo_unc

# Ministral SFT
CUDA_VISIBLE_DEVICES=6 python train_ministral_sft.py --epochs 3 --lr 2e-4

# Uncertainty Head
CUDA_VISIBLE_DEVICES=6 python train_uncertainty_head.py --epochs 50 --lr 1e-3
```
