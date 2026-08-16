# 제출 코드 — 2026 인하 AI 챌린지

첫 프레임 1장 + 16스텝 액션 궤적 → 16프레임 영상. **GPU 1장에서 백본부터 제출 CSV까지 통째로 돕니다.**

| | 점수 (낮을수록 좋음) |
|---|---:|
| 정적 기준선 (첫 프레임 16회 반복) | 0.3019 |
| 이 코드의 산출물 (`submission-bg.csv`) | **0.2091** |
| 배경 고정 후처리 없이 (`submission-raw.csv`) | 0.2299 |

---

## 1. 돌리는 법

```bash
export FT_ROOT=$HOME/ft          # 아래 4절 배치도대로 데이터·백본을 놓은 곳
export KIT_ROOT=$FT_ROOT/kit     # 생략하면 $FT_ROOT/kit
bash scripts/run_all.sh          # 학습 26시간 + 추론 52분
```

학습과 추론을 따로 돌리려면:

```bash
bash scripts/train.sh                    # 백본 → 9,000스텝
bash scripts/infer.sh                    # 마지막 체크포인트로 216문제 생성 → CSV
bash scripts/infer.sh /경로/가중치.ckpt   # 특정 체크포인트로
```

## 2. 규정 대조

| 항목 | 제한 | 실측 | |
|---|---|---|---|
| 학습 시간 | RTX PRO 6000 1장, 4일 | 9,000스텝 × 10.3초 ≈ **26시간** | 여유 |
| 추론 시간 | RTX PRO 6000 1장, 1시간 | 생성 49분 + 후처리 1.5분 + CSV 1분 ≈ **52분** | **여유 8분** |
| 외부 데이터 | 불가 | 대회 제공 학습셋만 사용 | |
| 평가셋 학습 | 금지 | `data/eval` 은 추론에서만 읽음 | |
| 채점 킷 수정 | 실격 | **한 줄도 고치지 않았습니다.** 킷 함수를 밖에서 부르기만 합니다 | |
| 사전학습 가중치 | 공개 + 허용 라이선스 | 대회가 제공한 `backbone.ckpt` 외 **없음** | |

**추론 여유가 8분뿐입니다.** DDIM 50스텝에 액션 CFG 2.0(순전파 2회)이라 이보다 무거워질 수 없습니다.
GPU 가 더 느린 환경이라면 `configs/infer.yaml` 의 `unconditional_guidance_scale` 을 1.0 으로 낮추면
순전파가 절반이 되어 약 27분에 끝납니다(점수는 0.2299 → 0.2306 수준으로 나빠집니다).

## 3. 방법 — 무엇이 점수를 만들었나

효과가 확인된 순서대로입니다. 괄호는 그 항목 하나를 넣었을 때의 점수 변화입니다.

**① 백본을 실제로 쓴다 (−0.21).** 대회 기본 설정은 제공된 `backbone.ckpt` 안의 1.44B UNet 을
쓰지 않고 32채널 11M 모델을 맨바닥부터 학습하도록 돼 있습니다. UNet 구조값을 백본 실측치에
맞추고(`model_channels 320` · `channel_mult [1,2,4,4]` · `num_head_channels 64` ·
`attention_resolutions [4,2,1]` · `use_scale_shift_norm False`) `only_reload_modules` 에 `model` 을
넣으면 1,468/1,517(96.8%)이 실립니다.

**② 액션을 cross-attention 으로 넣는다.** 액션 12차원(절대 6축 + 차분 6축)을 두 갈래로 만듭니다
(`src/model.py: ActionCtxUNet`).
- **프레임별 토큰 16개** — 이미지 토큰 자리에 더합니다. 프레임 t 에는 t 시점의 액션이 들어갑니다.
- **궤적 요약 토큰 32개** — 빈 텍스트 슬롯 77칸 중 뒤 32칸에 더합니다. 16프레임 전체가 공유합니다.

**③ 학습 스텝에는 정점이 있다 (−0.046).** 누적 9,000스텝이 최적입니다. 그 뒤로는 계속 나빠져
13,200스텝에서 정적 기준선(0.3019) 수준으로 되돌아갑니다. **더 오래 학습하면 안 됩니다.**

**④ 액션 CFG 2.0 (−0.011).** CFG(classifier-free guidance)는 조건을 준 예측과 안 준 예측의 차이를
증폭하는 추론 기법입니다. 2.0 이 최적이고 3.0 은 과증폭입니다(+0.006).

**⑤ 배경 고정 후처리 (−0.021).** 생성 영상에서 **시간에 따라 변한 자리(=로봇 팔)만 남기고**
나머지 픽셀은 첫 프레임으로 되돌립니다(`src/bgfreeze.py`). 학습도 GPU 도 필요 없고 216개에 90초입니다.
확산 모델이 배경에 남기는 미세한 흔들림이 DINO·Video 항목에서 그대로 감점되기 때문에 효과가 큽니다.

**학습이 4단계로 쪼개져 있는 이유**는 실제로 그렇게 학습했기 때문입니다. 단계를 넘을 때
UNet 가중치만 물려받고 **옵티마이저 상태는 새로 시작하며 학습률 워밍업도 다시 돕니다.**
9,000스텝을 한 번에 도는 것과 결과가 같지 않으므로, 재현하려면 이 순서 그대로여야 합니다.

| 단계 | 스텝 | 바뀌는 것 |
|---|---:|---|
| stage1 | 300 | 인코더 동결 · 액션 조건 **없음** — 백본을 데이터에 적응만 시킵니다 |
| stage2 | 2,100 | 인코더 해제 + 8비트 AdamW + **액션 cross-attention 투입** |
| stage3 | 1,800 | stage2 와 설정 동일 |
| stage4 | 4,800 | stage2 와 설정 동일 → 누적 **9,000** |

## 4. 파일 배치

```
$FT_ROOT/
  checkpoints/backbone.ckpt          대회 제공 사전학습 가중치
  data/train/                        학습 데이터 + so100_{action,delta}_statistics.json
  data/eval/                         평가 216문제
  outputs/                           학습 산출 (스크립트가 만듭니다)
  out/                               생성 영상 (스크립트가 만듭니다)
$KIT_ROOT/
  baseline/challenge_kit/            대회 킷 — 원본 그대로
  baseline/shared_libs/video_utils/
  submission_kit/make_submission_csv.py
```

## 5. 이 저장소

```
configs/   stage1~4.yaml  학습 4단계
           infer.yaml     추론 (stage4 설정 + CFG 2.0)
src/       model.py       액션 cross-attention UNet · SDPA 어텐션 · 동결 범위 · 8비트 Adam
           data12.py      12차원 액션(절대 6 + 차분 6) 데이터 모듈
           motion.py      움직임 가중 손실 (최종 레시피에서는 motion_weight 0.0 으로 미사용)
           generate.py    216문제 생성
           bgfreeze.py    배경 고정 후처리
scripts/   env.sh train.sh infer.sh run_all.sh
```

우리 코드는 이 5개 파일 816줄뿐이고, 나머지는 전부 대회 킷을 그대로 부릅니다.

## 6. 알아둘 점

- **`use_ema: False` 입니다.** 켜면 EMA 사본이 5.4GB 를 더 먹고, 백본에 `model_ema.*` 가 없어
  학습 0스텝 상태에서는 전 프레임이 노이즈로 나옵니다.
- **어텐션을 torch SDPA 로 교체합니다**(`src/model.py`). 킷 기본 구현으로는 32GB GPU 에 안 들어갑니다.
  96GB 환경에서는 없어도 되지만, 결과를 바꾸지 않으므로 그대로 둡니다.
- **`save_top_k: -1`, monitor 없음.** 검증 손실이 낮을수록 점수가 좋아지지 않았습니다(오히려 반대인
  구간이 있었습니다). 그래서 손실로 체크포인트를 고르지 않고 전부 남긴 뒤 제출 점수로 골랐습니다.
- **`submission-raw.csv` 도 같이 나옵니다.** 영상 후처리가 규정상 불가하다는 판단이 있을 경우를
  대비한 배경 고정 없는 판입니다(0.2299).
