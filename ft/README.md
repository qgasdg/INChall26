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

1. **한 번에 하나만 바꾼다.** 두 개를 같이 바꾸면 어느 쪽이 원인인지 영영 모른다.
2. **기준선은 step0(0.3031)로 고정.** 모든 비교는 여기에 댄다.
3. **결과는 [results/submissions.csv](../results/submissions.csv) 장부에 append**하고,
   아래 실험 표에도 한 줄 남긴다.
4. **제출은 하루 10회.** 변경 하나 = 제출 하나이므로 순서가 곧 예산이다.
   아래 우선순위는 그 예산 배분이다.

## 손잡이 목록

`open/baseline/challenge_kit/configs/train/inha_action_diffusion_11M.yaml` 과
`libs/dynamicrafter/lvdm/modules/networks/openaimodel3d.py` 를 읽고 뽑았다.
**"왜 의심되나"가 없는 항목은 넣지 않았다.**

| # | 손잡이 | 현재값 | 왜 의심되나 | 우선 |
|---|---|---|---|---|
| A | `base_learning_rate` | `1e-4` | **11M을 맨땅에서 학습시키려고 잡은 값이다.** 1.4B 사전학습 UNet을 이 lr로 미세조정하면 사전학습 지식이 초반에 날아간다(catastrophic forgetting). "0스텝이 최고"와 정확히 일치하는 증상 | ★★★ |
| B | `add_act_time_emb` | 없음 → **False** | False면 `time_embed` 출력이 절반(`embed_dim//2`)으로 줄고 거기에 액션을 **이어붙인다**(concat). 사전학습 ResBlock이 기대하는 emb 의미와 어긋난다. True면 같은 차원으로 **더한다**(addition) → 사전학습 호환. `openaimodel3d.py:419-434, 744-747` | ★★★ |
| C | 액션 표현 | **절대 관절값**, 전역 z정규화 | `lerobot_so100.py:388-396`. 16프레임 창 안에서 절대값은 거의 안 변한다 → 정규화하면 "지금 어느 자세대"만 남고 **"어떻게 움직이는지"는 신호가 거의 없다.** 셔플 액션이 정답 액션보다 나았던 관측과 맞물린다 | ★★★ |
| D | 유효 배치 | `batch_size 1` × `accumulate 2` = **2** | 1.4B 모델을 배치 2로 미세조정하면 gradient가 극도로 시끄럽다. A와 겹쳐 작용하면 파괴적 | ★★ |
| E | `action_dropout_prob` | `0.0` | 학습 때 액션을 안 지우면 추론에서 **액션 CFG**(액션을 준 예측과 안 준 예측을 비교해 액션 효과를 증폭하는 기법)를 못 쓴다. eval config도 `unconditional_guidance_scale: 1.0`(꺼짐) | ★★ |
| F | `precision: 16` | fp16 | `parameterization: "v"` + `rescale_betas_zero_snr` 조합에서 fp16은 불안정하기로 알려져 있다. bf16이 가능한 장비면 바꿀 값 | ★★ |
| G | `traj_len 16` / `downsample 1` / `fs_condition false` | — | 학습 데이터 fps와 대회 출력 fps(6fps 2.5초)가 어긋나면 모델이 배우는 프레임 간 이동량 자체가 틀어진다. `default_fs: 10`인데 조건화는 꺼져 있다 | ★★ |
| H | `use_ema: True` + `save_only_unet` | — | 저장된 체크포인트가 EMA 가중치인지 원가중치인지 확인 필요. 잘못 집으면 학습 곡선과 제출 점수가 따로 논다 | ★ |
| I | `target 320×512`, `pad: True` | — | eval 원본은 640×480. 패딩·리사이즈가 평가 시 어떻게 되돌려지는지 | ★ |

**이미 해결된 것 (다시 파지 말 것)**

- `only_reload_modules`에 `model`(UNet)이 빠져 있어 backbone.ckpt의 1.44B UNet을 안 쓰고
  32채널 11M을 맨땅에서 학습하던 문제 → 태양님이 해결. `0.5166 → 0.3031`.

## 실험 표

| ID | 바꾼 것 | 값 | 로컬 관측 | 제출 점수 | 판정 |
|---|---|---|---|---|---|
| — | (기준선) 태양 step0 | — | — | **0.3031** | 기준 |

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
