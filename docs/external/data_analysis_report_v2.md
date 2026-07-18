# 2026 인하 인공지능 챌린지 — 데이터 분석 리포트 v2 (EDA #01 개정)

작성일: 2026-07-16 · v1 대비 변경: **공식 평가 산식 반영**(claude/competition-rules.md), 규정 기반 전략 수정, SmolVLA 데이터셋 겹침 정량 확인 추가

---

## 1. 결론 요약

1. **태스크**: 현재 프레임 1장(640×480 PNG) + 미래 16스텝 행동(16×6 npy) → **16프레임(6fps) 미래 영상 생성**. 평가 샘플 216개 (Public 30% / Private 70%).
2. **공식 평가 산식** (낮을수록 좋음, 0이 최선):

   **Score = 0.3 × DINO + 0.3 × Video Feature + 0.4 × Action**

   - DINO / Video Feature: 제출킷이 추출한 feature와 정답 feature 간 **Cosine Distance**
   - Action: 생성 영상에서 역추정한 행동 vs 기준 행동의 **MAE**
   - **Action 가중치 0.4로 단일 성분 최대** → 행동 충실도가 최우선 순위
3. **train 데이터는 100% 공개 데이터**: HF Hub의 LeRobot SO-100 커뮤니티 데이터셋 128개를 6fps로 다운샘플한 것. **이 중 최소 101개(79%)가 SmolVLA 논문(arXiv:2506.01844) 부록 A.1의 사전학습 데이터셋 목록과 겹침을 직접 대조로 확인** (부록 목록 일부 페이지만 조회한 결과라 실제 겹침은 더 클 가능성).
4. **단, 규정상 외부 데이터 사용 불가** → 원본 HF 데이터(30fps·멀티뷰) 추가 사용은 **금지**. 반면 **라이선스가 허용된 공개 가중치 사전학습 모델은 사용 가능** (MIT/Apache 2.0/CC BY/CC BY-NC 등).
5. **action[t] ≈ state[t+1]** (평균 오차 1.65°, 상관 0.99+): action은 리더암 목표각으로 미래 팔 포즈를 거의 결정. 월드 모델의 불확실성은 "물체 반응"에 집중됨.
6. 확인한 eval 이미지 4장은 스팟 체크한 train 장면과 불일치 → **미공개 장면 포함 추정**(전수 대조 아님). 장면 암기보다 action-일반화가 필요.

---

## 2. 데이터 구성

```
open (1)/
├── data/
│   ├── train/                  # 128개 LeRobot 데이터셋 (user/dataset 구조)
│   │   ├── <user>/<dataset>/
│   │   │   ├── data/chunk-000/episode_XXXXXX.parquet   # 프레임 단위 테이블
│   │   │   ├── meta/info.json, episodes.jsonl          # 스키마·태스크 설명
│   │   │   └── videos/chunk-000/<camera_key>/episode_XXXXXX.mp4
│   │   └── so100_action_statistics.json                # action 정규화 mean/std
│   └── eval/
│       ├── images/sample_000000~000215.png             # 시작 프레임 (640×480)
│       └── actions/sample_000000~000215.npy            # (16, 6) float32, deg 단위
├── baseline/        # DynamiCrafter 기반 action-conditioned latent video diffusion (11M UNet)
└── submission_kit/  # 생성 mp4 → feature CSV 변환 (평가 feature 추출기 포함, 변경 금지)
```

### 전체 규모 (128개 데이터셋 meta 전수 집계)

| 항목 | 값 |
|---|---|
| 데이터셋 수 | 128 (HF repo ID 그대로) |
| 총 에피소드 / 프레임 | 11,132 / 1,025,666 (6fps 기준) |
| 로봇 | so100 125, so100-blue 2, so100-red 1 (전부 SO-100 계열 6DoF) |
| fps | 6fps 127개 / 10fps 1개 (원본 30fps, 1개는 50fps, stride 5 다운샘플) |
| 해상도 | 480×640 123개, 1080×1920 3개, 720×1280 2개 |
| 카메라 | 데이터셋당 1개 키만 유지 |
| 에피소드 길이 | 17~1,072 프레임 (중앙값 ~104 ≈ 17초, 첫 에피소드 기준) |

`info.json`의 `original_fps`, `downsample_stride`는 LeRobot 표준이 아닌 주최측 전처리 흔적.

---

## 3. Feature 상세

parquet 각 행 = 한 프레임. 모든 데이터셋 공통:

| feature | dtype/shape | 의미 |
|---|---|---|
| `action` | float32 (6,) | 리더암 목표 관절각 [deg]: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| `observation.state` | float32 (6,) | 팔로워암 실측 관절각 (동일 6개) |
| `observation.images.image` | video 480×640×3 | 카메라 프레임 (h264 mp4) |
| `timestamp` | float32 | 에피소드 내 시간(초), 1/6s 간격 |
| `frame_index` / `episode_index` / `index` / `task_index` | int64 | 구조적 인덱스 (`index`는 5 간격 = 원본 30fps 프레임 번호) |

- `episodes.jsonl`: 에피소드별 자연어 태스크 (pick-and-place 위주, 체스·물따르기·발치 등 다양. 일부 부실: "test", 빈 문자열)
- `so100_action_statistics.json`: count 974,661 ≈ 전체의 95% → **val 5% 제외한 train split 통계**로 추정. 모델·평가기 공통 정규화 기준.
- eval action은 **비정규화(deg)** 상태로 제공 → 입력 시 위 통계로 정규화 필요.

---

## 4. Feature 간 관계 (30개 데이터셋 × 첫 에피소드, 4,408프레임 표본)

**4-1. action ↔ state (리더-팔로워 구조, 핵심)**
같은 관절 corr 0.993~0.999 (gripper만 0.939). |action[t]−state[t]| = 2.25° > |action[t]−state[t+1]| = 1.65° → **action[t]은 사실상 state[t+1]의 예측치**. 팔의 미래 움직임은 action이 거의 결정하므로, 모델이 배워야 할 본질은 "액션→픽셀 렌더링 + 물체 상호작용".

**4-2. 관절 간 상관**: shoulder_lift↔elbow_flex **+0.89** (기구학적 커플링, 실효 자유도 < 6), wrist_flex↔gripper −0.33, shoulder_pan은 독립적.

**4-3. gripper 특수성**: 물체 파지 시 목표각-실측각 괴리 발생 → action-state 잔차가 파지 이벤트 신호. Action Component(0.4 가중치)에서 gripper 차원의 MAE 기여를 별도 모니터링할 가치.

**4-4. 분포**: 차원별 std 14~62로 스케일 격차 큼 → 정규화 필수. wrist_roll은 데이터셋 간 이질성 큼(카메라/마운트 세팅 차이).

---

## 5. 평가 산식 정밀 분석 (산식 확정 반영)

**Score = 0.3×DINO(cos dist) + 0.3×Video(cos dist) + 0.4×Action(MAE), 낮을수록 좋음.**

제출킷 코드와 결합하면 각 성분의 실체는:

| 성분 | 가중치 | feature | 거리 | 전략적 의미 |
|---|---|---|---|---|
| DINO | 0.3 | DINOv2 ViT-S/14 CLS, 프레임별 16×384 | GT feature와 cosine distance | 프레임 단위 시각 유사성. 배경·물체 외형을 GT와 비슷하게 유지해야 함 |
| Video Feature | 0.3 | R3D-18(Kinetics) 512-d 클립 feature | GT feature와 cosine distance | 시공간(모션) 유사성. 움직임 패턴이 맞아야 함 |
| Action | 0.4 | 고정 action extractor가 영상에서 역추정한 action | GT action(정규화)과 MAE | **가장 큰 가중치.** 액션 조건을 픽셀로 정확히 구현해야 함 |

핵심 통찰 (산식에서 도출):

1. **분포 거리(FVD)가 아니라 샘플별 feature 매칭이다.** cosine distance는 각 샘플의 GT 영상 feature와 1:1 비교 → "그럴듯한 영상"이 아니라 "그 장면의 GT와 유사한 영상"이 필요. 다양성·사실감보다 **정확도·일관성**이 유리하고, diffusion의 샘플 다양성은 오히려 리스크(고정 시드, 낮은 온도/eta, guidance 튜닝 필요).
2. **Action 0.4 + action extractor 고정(변경 금지)** → extractor가 인식하기 좋은 형태로 "팔의 움직임을 선명하게" 렌더링하는 것이 점수의 40%를 좌우. 로컬에서 extractor로 자가 채점하는 루프가 사실상 필수.
3. eval GT 영상은 서버에만 있으므로, 로컬 프록시는 train 홀드아웃으로 (첫 프레임 + 16 action) → 생성 → 제출킷 feature → GT 영상 feature와 cosine/MAE 계산으로 동일 산식 재현 가능.
4. Public 30%(65샘플 수준)라 노이즈 큼 + 하루 3회 제출 제한 + **최종 채점 파일 1개 수동 선택** → Public 과적합 금지, 로컬 지표 신뢰 우선.

### 규정에 따른 제약 (전략에 직접 영향)

- **외부 데이터 금지**: HF 원본(30fps·멀티뷰) 추가 활용 불가. v1 리포트의 해당 전략은 **폐기**.
- **eval 데이터 학습 활용 금지** (pseudo-labeling 포함).
- **사전학습 모델은 공개 가중치 + 허용 라이선스면 사용 가능** → DynamiCrafter, SVD, DINOv2, VLM 등 활용 여지. 각 모델 라이선스 확인 필수.
- **컴퓨트 제한**: 학습 ≤ 4일, 추론 ≤ 1시간 (RTX PRO 6000 96GB 1장) → 216샘플 × 16프레임 생성을 1시간 내 = 샘플당 평균 ~16.6초. 무거운 샘플링(스텝 수)·앙상블 설계 시 시간 예산 계산 필수.
- 제출킷 변경 금지, CSV 후처리 금지.

---

## 6. 공개 데이터셋 출처 (v1 결론 + 정량 보강)

**결론: train 128개 전부 HF 공개 LeRobot SO-100 커뮤니티 데이터셋.** 근거:

1. 폴더명 = HF repo ID, LeRobot v2.0/v2.1 표준 포맷.
2. HF 공식 블로그 [LeRobot Community Datasets](https://huggingface.co/blog/lerobot-datasets)가 동일 repo ID 3개를 직접 예시로 언급.
3. **신규(정량)**: SmolVLA 논문 부록 A.1의 481개 사전학습 데이터셋 목록과 대조 → **128개 중 최소 101개(79%) 일치** (부록 목록의 일부 페이지만 조회했으므로 하한값. 나머지 27개도 같은 커뮤니티 생태계 데이터로 HF에서 개별 확인 가능, 예: [samsam0510/tooth_extraction_3](https://huggingface.co/datasets/samsam0510/tooth_extraction_3), [triton7777/so100_dataset_mix](https://huggingface.co/datasets/triton7777/so100_dataset_mix)).
4. 원본 열람: `https://huggingface.co/datasets/<user>/<dataset>` + [시각화 뷰어](https://lerobot-visualize-dataset.hf.space/lirislab/close_top_drawer_teabox/episode_25).

대회본 vs 원본 차이: 6fps 다운샘플, 카메라 1개만 유지, 메타 축소. **원본 추가 사용은 규정상 금지**이므로 출처 확인의 실익은 (a) 데이터 특성·수집 방식 이해, (b) 동일 데이터를 쓴 선행연구(SmolVLA 등)의 전처리·필터링 노하우 차용에 있음.

### 관련 논문/연구

| 구분 | 자료 | 링크 |
|---|---|---|
| 데이터 생태계 소개 | LeRobot Community Datasets 블로그 | https://huggingface.co/blog/lerobot-datasets |
| 포맷/라이브러리 | LeRobot 라이브러리 · 논문 | https://github.com/huggingface/lerobot · https://arxiv.org/abs/2602.22818 |
| **동일 데이터 사용 대표 논문** | **SmolVLA** (481개 커뮤니티 데이터셋으로 VLA 사전학습, 본 대회 데이터와 ≥79% 겹침 확인) | https://arxiv.org/abs/2506.01844 |
| 하드웨어 | SO-ARM100 | https://github.com/TheRobotStudio/SO-ARM100 |
| 베이스라인 백본 | DynamiCrafter | https://arxiv.org/abs/2310.12190 |
| 평가 feature | DINOv2 · R3D-18 · cdfvd(로컬 평가용) | https://arxiv.org/abs/2304.07193 · https://arxiv.org/abs/1711.11248 · https://arxiv.org/abs/2404.12391 |

---

## 7. 전략 우선순위 (산식 반영 개정)

1. **[최우선] Action fidelity (0.4)**: action 조건 주입 강화(채널 concat + AdaLN/cross-attn 등 비교), 로컬 action-extractor 자가 채점 루프 구축.
2. **[평가 재현] 로컬 리더보드**: train 홀드아웃에서 산식 그대로(0.3/0.3/0.4, cosine/MAE) 재현 — 제출 3회/일 제한 대응.
3. **[샘플별 매칭 최적화]**: cosine distance는 1:1 비교 → 결정론적·저분산 생성 세팅(ddim eta, guidance, 시드)을 로컬 지표로 튜닝. "예쁜 영상"보다 "GT에 가까운 영상".
4. **[도메인 갭] eval 미공개 장면 대응**: 다양한 데이터셋 균형 샘플링, 색/조명 증강 (외부 데이터 없이).
5. **[품질 필터링]**: SmolVLA식 큐레이션(임베디먼트/에피소드 수/품질/프레임 커버리지 기준) 차용해 128개 중 저품질 제외 실험.
6. **[컴퓨트 예산]**: 추론 1시간/216샘플 → 생성 스텝 수·해상도 트레이드오프 사전 계산.

## 8. 한계

- 상관 분석은 128개 중 30개 데이터셋 첫 에피소드 표본(4,408프레임).
- eval-train 장면 대조는 8개 데이터셋 스팟 체크.
- SmolVLA 목록 대조는 부록 일부 페이지 기준(101/128은 하한).
- cosine distance의 서버측 세부(프레임별 평균인지, 벡터 연결 후 1회인지)는 미공개 — 로컬 재현 시 두 방식 모두 계산해 보수적으로 판단할 것.
