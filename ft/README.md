# ft — baseline 파인튜닝, 한 칸씩

담당: 윤택. 태양님의 DynamiCrafter 결합(step0 0.3031)을 **출발선으로 삼고**,
설정을 하나씩만 바꿔가며 점수가 어디서 움직이는지 찾는다.

## 왜 이 방향인가

제출 이력([docs/dacon_submission.md](../docs/dacon_submission.md))이 이상하다.

```
0스텝    0.3031  ← 최고
500스텝  0.3270
1250스텝 0.3467
1750스텝 0.3292
```

**학습할수록 나빠진다.** 그런데 리더보드 1위는 0.171이고 10위도 0.248이다.
정적(0.3019)을 크게 이기는 팀이 열 팀 넘는다 — 즉 **움직임이 손해인 판이 아니라
우리 학습이 잘못 굴러가고 있는 것**이다. 원인이 레시피 어딘가에 있고,
아직 아무도 안 건드린 손잡이가 남아 있다는 가정에서 출발한다.

## 원칙

1. **결과는 [results/submissions.csv](../results/submissions.csv) 장부에 append**하고,
   아래 실험 표에도 한 줄 남긴다.
2. 데이콘에 결과를 제출할 때마다 커밋/푸시로 코드 기록을 남겨 놓는다.

## 손잡이 목록

`open/baseline/challenge_kit/configs/train/inha_action_diffusion_11M.yaml` 과
`libs/dynamicrafter/lvdm/modules/networks/openaimodel3d.py` 를 읽고 뽑았다.
**"왜 의심되나"가 없는 항목은 넣지 않았다.**

| # | 손잡이 | 현재값 | 왜 의심되나 | 우선 |
|---|---|---|---|---|
| A | `base_learning_rate` | `1e-4` | **11M을 맨땅에서 학습시키려고 잡은 값이다.** 1.4B 사전학습 UNet을 이 lr로 미세조정하면 사전학습 지식이 초반에 날아간다(catastrophic forgetting). "0스텝이 최고"와 정확히 일치하는 증상 | ★★★ |
| B | `add_act_time_emb` | 없음 → **False** | False면 `time_embed` 출력이 절반(`embed_dim//2`)으로 줄고 거기에 액션을 **이어붙인다**(concat). 사전학습 ResBlock이 기대하는 emb 의미와 어긋난다. True면 같은 차원으로 **더한다**(addition) → 사전학습 호환. `openaimodel3d.py:419-434, 744-747` | ★★★ |
| C | 액션 표현 | **절대 관절값**, 전역 z정규화 | `lerobot_so100.py:388-396`. 창 안에서도 팔은 꽤 움직인다(창내 std가 전역의 32.6%). 문제는 비율이다 — **모델이 받는 값 변동의 약 88%가 "어느 자세대인가"고 "이 창 안에서 무슨 일이 일어나는가"는 12%뿐**이며, 프레임 간 변화량은 0.079로 값 크기의 8% 수준이다. 손실을 줄이는 쉬운 길은 큰 성분(자세대→화면 배치)이라 움직임 성분은 gradient 를 거의 못 받는다. 셔플 액션이 정답보다 나았던 관측과 맞물린다 (`ft/probe_action_signal` 실측) | ★★★ |
| D | 유효 배치 | `batch_size 1` × `accumulate 2` = **2** | 1.4B 모델을 배치 2로 미세조정하면 gradient가 극도로 시끄럽다. A와 겹쳐 작용하면 파괴적 | ★★ |
| E | `action_dropout_prob` | `0.0` | 학습 때 액션을 안 지우면 추론에서 **액션 CFG**(액션을 준 예측과 안 준 예측을 비교해 액션 효과를 증폭하는 기법)를 못 쓴다. eval config도 `unconditional_guidance_scale: 1.0`(꺼짐) | ★★ |
| F | `precision: 16` | fp16 | `parameterization: "v"` + `rescale_betas_zero_snr` 조합에서 fp16은 불안정하기로 알려져 있다. bf16이 가능한 장비면 바꿀 값 | ★★ |
| G | `traj_len 16` / `downsample 1` / `fs_condition false` | — | 학습 데이터 fps와 대회 출력 fps(6fps 2.5초)가 어긋나면 모델이 배우는 프레임 간 이동량 자체가 틀어진다. `default_fs: 10`인데 조건화는 꺼져 있다 | ★★ |
| H | `use_ema: True` + `save_only_unet` | — | 저장된 체크포인트가 EMA 가중치인지 원가중치인지 확인 필요. 잘못 집으면 학습 곡선과 제출 점수가 따로 논다 | ★ |
| I | `target 320×512`, `pad: True` | — | eval 원본은 640×480. 패딩·리사이즈가 평가 시 어떻게 되돌려지는지 | ★ |

**이미 해결된 것 (다시 파지 말 것)**

- `only_reload_modules`에 `model`(UNet)이 빠져 있어 backbone.ckpt의 1.44B UNet을 안 쓰고
  32채널 11M을 맨땅에서 학습하던 문제 → 태양님이 해결. `0.5166 → 0.3031`.

## 백본을 제대로 싣기까지 고친 것 (2026-08-07)

대회 config 그대로는 **1,107개 텐서 중 1개**만 실린다. 아래를 다 맞춰야 1.4B 가 온전히 올라간다.
전부 `backbone.ckpt`(= HF `Doubiiu/DynamiCrafter_512` 의 `model.ckpt`) 실측 shape 으로 확정했다
(`probe_backbone.py`, `probe_load.py`).

| 항목 | 대회 원본 | 실측 | 안 고치면 |
|---|---|---|---|
| `model_channels` | 32 | **320** | 구조 자체가 다름 |
| `channel_mult` | [1,2,3] | **[1,2,4,4]** | 〃 |
| `num_head_channels` | 16 | **64** | 〃 |
| `attention_resolutions` | [4,2] | **[4,2,1]** | 〃 |
| `use_scale_shift_norm` | True | **False** | ResBlock `emb_layers` 가 정확히 2배 어긋나 적재 자체가 예외로 죽는다 |
| `add_act_time_emb` | 없음(=False) | **True** | `time_embed.2` 가 shape 불일치로 랜덤 초기화 |
| `only_reload_modules` | UNet 제외 | **UNet 포함** | 백본이 아예 안 실림 |

**★`use_ema: True` — 0스텝 생성에서 치명적.**
`LitEma` 는 모델을 만드는 시점의 가중치를 복사해두는데 그 시점은 **사전학습을 싣기 전**이고,
`backbone.ckpt` 에는 `model_ema.*` 가 없어 갱신되지도 않는다. 그래서 `ema_scope()` 안에서
생성하면 **랜덤 가중치로 생성한다.** 실제로 전 프레임이 컬러 노이즈로 나왔고, 끄자 정상 영상이 나왔다.
학습된 체크포인트에는 EMA 가 저장되므로 학습 후 추론에서는 드러나지 않는다 — 0스텝에서만 터진다.

`action_embed` 를 0으로 초기화해도 노이즈는 그대로였다 → **액션 임베딩은 원인이 아니었다.**

## 실험 표

| ID | 바꾼 것 | 값 | 로컬 관측 | 제출 점수 | 판정 |
|---|---|---|---|---|---|
| — | (기준선) 태양 step0 | — | — | **0.3031** | 기준 |
| ft-01 | 백본 적재 정정(7군데) + `use_ema` off + 액션 12차원 | 0스텝 | 앞 8프레임 양호 · 뒤 8프레임 붕괴 | **0.3729** | ✗ 기준선보다 0.0698 나쁨 (코드 `09e75e5`) |
| ft-05 | + `action_embed` zero-init · `fs_condition` true · `ddim_eta` 0 · `guidance_rescale` 제거 · DDIM 15 | 0스텝 | **붕괴 소멸** — 16프레임 내내 물체 위치 유지 | **0.3948** | ✗ 더 나빠짐 |

**ft-01 이 말해주는 것.**
백본을 96.8% 실은 0스텝이 태양님 step0(0.3031)보다 **0.0698 나쁘다.**
"백본만 제대로 실으면 된다"는 기대는 성립하지 않았다.

**단, 원인은 아직 특정 못 한다.** 태양님 config 를 못 봤으므로 두 실행의 정확한 차이를 모른다
(태양님도 1.4B 를 올렸으니 구조 파라미터는 이미 맞췄을 것이다 — "원본 config 로는 1개만 실린다"는
**원본** 이야기이지 태양님 이야기가 아니다). 우리가 한 번에 세 가지를 같이 바꾼 것도 문제다:
`use_ema` off · 액션 12차원 · `add_act_time_emb` True. **다음 실험은 이걸 쪼개야 한다.**

쪼개는 순서 제안 — 12차원은 액션 MLP 가 랜덤이라 0스텝에서 순수 잡음원이므로 이것부터 뺀다.

| 다음 | 바꿀 것 | 묻는 것 |
|---|---|---|
| ft-02 | 액션 **6차원**으로 되돌림 (나머지 동일) | 12차원이 깎아먹었나 |
| ft-03 | `action_conditioned: False` | 랜덤 액션 MLP 자체가 손해인가 |
| ft-04 | `parameterization`/`rescale_betas_zero_snr` 조합 | 뒤 8프레임 붕괴의 원인인가 |

## 실행

```bash
cd open/baseline/challenge_kit
bash scripts/train.sh --config ../../../ft/configs/<이름>.yaml --script scripts/train_diffusion.py
```

`ft/configs/`에 변경 하나당 yaml 하나를 둔다. 원본 config는 건드리지 않는다 —
**diff가 곧 실험 내용**이어야 하기 때문이다.

## 남은 약점

제출 점수 말고 품질을 잴 수단이 없다. 하루 10회로는 변수 열 개가 상한이고,
같은 설정을 두 번 내볼 여유가 없어 **노이즈인지 실제 개선인지 구분이 안 된다.**
A~C가 정리되면 자체 IDM(연속 두 프레임에서 관절 움직임을 역추정하는 모델) 학습으로
로컬 지표를 붙이는 것이 다음 순서다.
