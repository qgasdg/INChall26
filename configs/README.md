# `configs/` — 3계층 config

병합 우선순위(뒤가 우선): **`base.yaml` < `exp/<expNN>.yaml` < `server/<이름>.yaml` < CLI `--set`**.

- `base.yaml` — 공통 기본값(과제 규격·데이터·모델·학습·추론·평가 디폴트). 팀 확정 설계가 인코딩됨.
- `exp/` — **실험 1개 = 파일 1개 + 커밋 해시 1개.** base 위에 바뀌는 값만. 단일 변수 원칙. 하드웨어 값 금지.
- `server/` — 하드웨어 종속값만(batch/precision/경로/max_steps). 같은 exp를 서버 바꿔 재현.

## 사용
```bash
python -m src.cli config --exp configs/exp/exp-01_e3_lora.yaml --server configs/server/a6000.yaml
python -m src.cli config --exp configs/exp/exp-01_e3_lora.yaml --server configs/server/a6000.yaml --set train.lr=5e-5
```
로더: [`src/utils/config.py`](../src/utils/config.py) `load_config()`.

## 명명
- `exp/exp-01_e3_lora.yaml`, `exp/exp-02_ft_full.yaml` … (exp_id = docs/13 통합 원장, ladder = E0~E8)
- `server/a6000.yaml`, `server/runpod_4090.yaml`, `server/v100.yaml` …
