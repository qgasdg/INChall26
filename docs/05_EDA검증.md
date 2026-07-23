# EDA 검증 리포트 — 실데이터로 노션 리포트 대조 (2026-07-18)

> **EDA(탐색적 데이터 분석)** = 데이터를 이리저리 뜯어보며 성질을 파악하는 작업. 이 문서는 노션 리포트의 주장을 실제 데이터로 직접 확인한 결과다.

> 스크립트: [eda01_verify.py](../scripts/eda01_verify.py), [eda02_action_state.py](../scripts/eda02_action_state.py) (`.venv/bin/python`)
> 대상: open.zip (8.6GB, 24,016파일) → `open/` 압축 해제본

## 결론: 노션 데이터 분석 리포트 v2의 주장 전부 실측 일치. 신규 발견 4건.

## 1. 전수 집계 — 완전 일치
데이터셋/에피소드/프레임 수, fps·해상도·로봇 분포, eval 규격, action stats까지 노션 수치와 전부 일치 (세부 수치는 [03_데이터분석.md](03_데이터분석.md) §2 참조, 재실행은 `scripts/eda01_verify.py`).

## 2. action(명령)↔state(실제 자세) 관계 — 성질 결론 재현 (표본이 달라 수치는 근소 차이)
무작위 30개 데이터셋 첫 에피소드 2,711프레임 (노션은 다른 30개 표본 4,408프레임):
- `|이번명령 − 이번자세|` = **2.96°** > `|이번명령 − 다음자세|` = **1.99°** (노션 2.25/1.65) → **이번 명령 ≈ 다음 순간 자세 확정** ✅
- 상관(이번명령, 다음자세): 어깨좌우 .999 / 어깨상하 .993 / 팔꿈치 .977 / 손목굽힘 .998 / 손목회전 .9995 / **집게 .951 (최저)** ✅
- 어깨상하↔팔꿈치 상관 +0.78 (노션 +0.89 — 표본 차이, 맞물림 방향은 일치) ✅
- 관절별 오차(다음자세 기준) 기여: **어깨상하 3.0° / 팔꿈치 3.2° / 집게 3.1°가 대부분**, 어깨좌우·손목회전은 0.7~0.8°

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
1. **`action_dropout_prob: 0.0`** — 베이스라인은 **"행동 없음" 경로를 학습한 적이 없음**(학습 중 행동을 일부러 빼는 비율이 0). 텍스트만 5% 빼고 학습됨 → 행동 기반 CFG(행동을 얼마나 세게 따를지 조절)를 하려면 "행동 없음" 경로를 새로 학습하거나(행동 dropout>0) 텍스트 기반 CFG만 가능. E1 설계 시 결정적 제약.
2. **16프레임 미만 에피소드 34개(0.31%)** — 이 짧은 것들로는 16장 구간을 못 만듦. 에피소드 최소 길이=1프레임(전수 집계 기준, 노션의 17~1,072는 첫 에피소드만 봐서 최소값이 달랐음). 데이터 로더에서 걸러내기 필수(크래시 방지).
3. **eval 행동 분포 vs train 통계 차이**: eval의 손목회전(wrist_roll) 평균 **+64.4°** vs train 평균 **−29.4°**(표준편차 62.6, 약 +1.5σ만큼 밀림), 어깨상하는 eval 85.9° vs train 117.6°. eval이 train 분포의 한가운데에서 벗어나 있음 → 처음 보는 장면용 홀드아웃 설계 시 손목회전 분포도 고려할 가치.
4. **카메라 키 예외 1개**: `liyitenga/so100_bi_giveme5`만 `observation.images.s_left`, 나머지 127개는 `observation.images.image`.

## 6. 영상 표본 점검 (8개)
코덱 h264 / 색저장 yuv420p / 초당 6장 정확, 프레임 수 61~115. 해상도는 640×480 외 1280×720도 있음(메타 정보와 일치) — 학습 전 320×512로 비율 유지 여백 채우기(letterbox)로 통일 필요.

## 다음 단계 제안
- E0 준비: 베이스라인 재현 환경 구축 (필요 패키지 확인, GPU 머신 확보 — 로컬 맥에서는 학습 불가)
- 로컬 리더보드 구축: 킷의 전처리 과정을 그대로 재현 (학습 홀드아웃의 정답 특징 추출)
- 노션 잔여 미확정 2건(서버가 코사인을 합치는 방식, sample_submission 출처)은 E0 제출 1회로 맞춰봄
