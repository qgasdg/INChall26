# EDA 검증 리포트 — 실데이터로 노션 리포트 대조 (2026-07-18)

> 스크립트: [eda01_verify.py](../scripts/eda01_verify.py), [eda02_action_state.py](../scripts/eda02_action_state.py) (`.venv/bin/python`)
> 대상: open.zip (8.6GB, 24,016파일) → `open/` 압축 해제본

## 결론: 노션 데이터 분석 리포트 v2의 주장 전부 실측 일치. 신규 발견 4건.

## 1. 전수 집계 — 완전 일치
데이터셋/에피소드/프레임 수, fps·해상도·로봇 분포, eval 규격, action stats까지 노션 수치와 전부 일치 (세부 수치는 [03_데이터분석.md](03_데이터분석.md) §2 참조, 재실행은 `scripts/eda01_verify.py`).

## 2. action↔state 관계 — 정성 결론 재현 (표본이 달라 수치 근소 차이)
무작위 30개 데이터셋 첫 에피소드 2,711프레임 (노션은 다른 30개 표본 4,408프레임):
- `|a[t]−s[t]|` = **2.96°** > `|a[t]−s[t+1]|` = **1.99°** (노션 2.25/1.65) → **action[t] ≈ state[t+1] 확정** ✅
- corr(a[t], s[t+1]): pan .999 / lift .993 / elbow .977 / wrist_flex .998 / wrist_roll .9995 / **gripper .951 (최저)** ✅
- shoulder_lift↔elbow_flex corr +0.78 (노션 +0.89 — 표본 차이, 커플링 방향 일치) ✅
- 관절별 MAE(t+1) 기여: **lift 3.0° / elbow 3.2° / gripper 3.1°가 지배**, pan·roll은 0.7~0.8°

## 3. submission_kit 코드 검증 — exp02 주장 전부 일치
- `make_submission_csv.py` 기본값: **320×512, pad(letterbox)=True, temporal_length=16** (16프레임 아니면 예외 발생) ✅
- DINO: timm **`vit_small_patch14_dinov2.lvd142m`**, 입력 크기는 모델 정의(518)로 자동 해석, **CLS 토큰**, 출력 (16,384) ✅
- Video: torchvision **r3d_18(Kinetics) fc=Identity**, `F.interpolate(trilinear, (T,112,112))` — **비등방 squish 확인**, 출력 (512,) ✅
- Action: **MAE를 킷이 로컬 계산** — `mean(|pred−target|)` → (1,1) 스칼라를 CSV에 기록. target은 eval npy를 so100 stats로 **z-score 후** 비교 ✅
- action_extractor 구조: Conv3D stem(stride 1,2,2) + Residual3D 스테이지(**temporal stride 없음**) → **Bi-GRU(256)** → MLP → (16,6) ✅
- `build_inference_batch`: `video_np[0] = 조건 이미지` — 조건 프레임 = 출력 0번 프레임 ✅
- sample_submission.csv: 3성분 × 216행, DINO (16,384) / Video (512,) / **Action mean 0.579 (0.088~1.148)** — 노션 수치 그대로 ✅

## 4. baseline config 검증 — exp03 주장 전부 일치
- eval: **ddim_steps 50, CFG=1.0(미사용), ddim_eta=1.0(확률적!), batch_size 1**, uniform_trailing, guidance_rescale 0.7(CFG=1이라 무효), camera_key `observation.images.top`(잔재) ✅ → **E1 무학습 개선 여지 재확인**
- train: v-prediction + zero-SNR, dynamic rescale, perframe_ae, in_channels 8, model_channels 32, action_conditioned(6dim), uncond_prob 0.05(empty_seq=텍스트), batch 1×accum 2, lr 1e-4, 100K steps/48h, precision 16, val 5% ✅

## 5. 신규 발견 (노션에 없던 것)
1. **`action_dropout_prob: 0.0`** — 베이스라인은 **action 무조건화 경로를 학습한 적이 없음**. 텍스트(empty_seq) uncond만 5% 학습됨 → action-CFG를 하려면 uncond 경로를 새로 학습하거나(action dropout>0) 텍스트-uncond 기반 CFG만 가능. E1 설계 시 결정적 제약.
2. **16프레임 미만 에피소드 34개(0.31%)** — 윈도우 생성 불가분. 에피소드 길이 min=1(전수 집계, 노션의 17~1,072는 첫 에피소드 기준이라 min이 달랐음). 데이터 로더에서 필터 필수.
3. **eval action 분포 vs train 통계 차이**: eval wrist_roll mean **+64.4°** vs train mean **−29.4°**(std 62.6, 약 +1.5σ), shoulder_lift eval 85.9° vs train 117.6°. eval이 train 분포의 중심에서 벗어나 있음 → unseen-scene 홀드아웃 설계 시 wrist_roll 분포도 고려 가치.
4. **카메라 키 예외 1개**: `liyitenga/so100_bi_giveme5`만 `observation.images.s_left`, 나머지 127개는 `observation.images.image`.

## 6. 영상 스팟체크 (8개)
h264 / yuv420p / 6fps 정확, 프레임 수 61~115. 해상도는 640×480 외 1280×720 존재(메타와 일치) — 학습 전 320×512 letterbox 통일 필요.

## 다음 단계 제안
- E0 준비: baseline 재현 환경 구축 (requirements 확인, GPU 머신 확보 — 로컬 맥에서는 학습 불가)
- 로컬 리더보드 구축: 킷 전처리 체인 재현 (train 홀드아웃 GT feature 추출)
- 노션 잔여 미확정 2건(서버 cosine 집계 방식, sample_submission 출처)은 E0 제출 1회로 캘리브레이션
