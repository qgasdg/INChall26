# base — 대회 baseline + 검증된 필수 정정만

대회가 준 `inha_action_diffusion_11M.yaml` 에서 **측정으로 검증된 것만** 고쳤다.
확산 생성 파이프라인은 그대로 쓴다(잔차·픽셀합성·대조손실 같은 실험적 장치는 넣지 않는다).

## 무엇을 고쳤나

전부 `backbone.ckpt`(= HF `Doubiiu/DynamiCrafter_512` 의 `model.ckpt`) 실측 shape 으로 확정했다.
**대회 원본 config 로는 1,107개 텐서 중 1개만 실린다** — 즉 1.44B 백본을 안 쓰고 32채널 11M 을
맨땅에서 학습한다. 아래를 다 맞추면 1,468/1,517(96.8%)이 실린다.

| 항목 | 대회 원본 | 여기 | 안 고치면 |
|---|---|---|---|
| `model_channels` | 32 | **320** | 구조가 다름 |
| `channel_mult` | [1,2,3] | **[1,2,4,4]** | 〃 |
| `num_head_channels` | 16 | **64** | 〃 |
| `attention_resolutions` | [4,2] | **[4,2,1]** | 〃 |
| `use_scale_shift_norm` | True | **False** | ResBlock `emb_layers` 가 정확히 2배 어긋나 **적재가 예외로 죽는다** |
| `only_reload_modules` | UNet 제외 | **UNet 포함** | 백본이 아예 안 실림 |
| `add_act_time_emb` | 없음(=False) | **True** | `time_embed` 출력이 1280→640 이 되어 사전학습 `(1280,1280)` 과 shape 충돌 |
| `action_dims` | 6 | **12** (절대 6 + Δ 6) | — |
| `fs_condition` | false | **true** | 사전학습 `fps_embedding` 을 버림 |
| `base_learning_rate` | 1e-4 | **1e-5** | 1e-4 는 11M 맨땅 학습용. 사전학습 미세조정에는 과하다 |
| `precision` | 16 | **bf16** | v-예측 + ZTSNR 에서 fp16 은 불안정 |
| `accumulate_grad_batches` | 2 | **16** | 1.4B 를 유효 배치 2 로 미세조정하면 gradient 가 너무 시끄럽다 |

## ★ 추론에서 `use_ema`

**학습 체크포인트 없이(0스텝) 생성할 때는 `--no-ema` 가 필수다.** `LitEma` 는 모델을 만드는
시점의 가중치를 복사해두는데 그 시점은 **사전학습을 싣기 전**이고, `backbone.ckpt` 에는
`model_ema.*` 가 없어 갱신되지도 않는다. 켜두면 **랜덤 가중치로 생성해 전 프레임이 컬러 노이즈**가 된다.
학습된 체크포인트에는 EMA 가 저장되므로 학습 후 추론에서는 드러나지 않는, 0스텝에서만 터지는 함정이다.

## 액션 12차원

`data12.py` 가 킷 데이터모듈을 감싸 `[T,6]`(절대값 z정규화) → `[T,12]`(절대 6 + 정규화된 Δ 6)로 만든다.
Δ 는 `make_delta_stats` 로 뽑은 Δ 전용 통계로 정규화한다 — 절대값 통계로 나누면 Δ 가 0 근처로 눌린다
(Δ std 3.3~5.8도 vs 절대 std 26~63도).

**추론(`generate.py:to_12dim`)과 학습(`data12.py`)이 같은 식이어야 한다.** 다르면 분포가 어긋난다.

## 파일

```
base-01.yaml   학습 설정
data12.py      12차원 액션 데이터모듈 (킷 원본 무수정)
generate.py    eval 영상 생성 (--ckpt 로 학습본 적재, 없으면 0스텝)
```

## 실행

```bash
# 학습
cd open/baseline/challenge_kit
export PYTHONPATH=<base 경로>:$(pwd)/src:$(pwd)/libs/dynamicrafter:$(pwd)/../shared_libs/video_utils
export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=12345
python scripts/train_diffusion.py --base <base 경로>/base-01.yaml --train --devices 1

# 생성 (0스텝)
python generate.py --config base-01.yaml --action-dims 12 --no-ema --limit 216 --out <출력>

# 생성 (학습본)
python generate.py --config base-01.yaml --action-dims 12 --ckpt <학습>.ckpt --limit 216 --out <출력>

# 제출 CSV — 공식 킷 무수정 (2026-08-03 공지상 허용된 유일한 용도)
cd open/submission_kit
python make_submission_csv.py --prediction-root <출력> --challenge-root <eval> \
  --action-stats-path <train>/so100_action_statistics.json --output-csv submission.csv
```
