# 작업 로그

## 2026-07-18
- 대회 파악 시작. 대회 URL 확인: https://dacon.io/competitions/official/236736
- 개요/규칙/평가/데이터 페이지 조사 → [01_대회개요.md](01_대회개요.md)에 정리
- 핵심 파악 사항:
  - 과제: 이미지 1장 + 행동 시퀀스 → 미래 로봇 영상 생성 (월드 모델)
  - 평가: 0.3·DINO + 0.3·VideoFeature + 0.4·Action, 낮을수록 좋음
  - 제약: 외부 데이터 금지, 허용 라이선스 사전학습 모델만, 학습 4일(RTX PRO 6000)/추론 1h
- 데이터(open.zip) 미보유 상태 — 사용자가 다운로드 진행 중 (시각 데이터라 용량 큼)
- 팀 노션(https://app.notion.com/p/2026-39f6af585380800e931de40bb79f6079) 전체 수집 → docs에 정리:
  - [02_평가산식_분석.md](02_평가산식_분석.md): cosine/MAE 수식 분석, 논문 검증, 성분별 전략, 트레이드오프
  - [03_데이터분석.md](03_데이터분석.md): EDA v2 — 128개 LeRobot SO-100 데이터셋(1.03M 프레임), action[t]≈state[t+1], 킷 추출기 실체(DINOv2 ViT-S/14·R3D-18·고정 IDM)
  - [04_파이프라인.md](04_파이프라인.md): v1→v1.3 — Wan2.2-TI2V-5B LoRA + frame-level adaLN 주력, E0~E8 실험 사다리, 조건 프레임 포함 16프레임 확정
- 01_대회개요.md 과제 정의 보정 (조건 프레임 포함 16프레임)
- 주요 발견: Action Component는 킷이 로컬에서 실측 계산 가능(CSV에 기록됨) / 베이스라인 추론이 η=1.0·CFG미사용이라 E1 cheap win 존재 / 팀 구성 전 1회 제출 필수(8/17 한)
- open.zip(8.6GB) 도착 → `open/`에 압축 해제, 프로젝트 venv(.venv) 구축 (numpy/pandas/pyarrow/opencv/matplotlib)
- **EDA 검증 완료** → [05_EDA검증.md](05_EDA검증.md): 노션 리포트 수치 전부 실측 일치 (128 데이터셋/11,132 에피소드/1,025,666 프레임, eval 216, action[t]≈state[t+1], 킷 스펙, baseline config)
- 신규 발견: ① baseline `action_dropout_prob=0` → action-CFG uncond 경로 미학습 (E1 제약) ② <16프레임 에피소드 34개 필터 필요 ③ eval wrist_roll 분포가 train 대비 +1.5σ 시프트 ④ 카메라 키 예외 1개(s_left)
- TODO: E0(baseline 재현+제출 1회) 준비 — GPU 머신 필요. 로컬 리더보드(킷 체인 재현) 구축.
- 운영 전략 수립 → [07_운영전략.md](07_운영전략.md): 2-머신 체제(Mac 개발/GPU 서버 학습), private repo + trunk-based, 노션=기획·repo=정본, 실험 트래킹 = repo CSV 정본(local_lb/submissions) + wandb 보조, expNN은 exp04부터. git init + .gitignore 완료. 최우선 미결정 = GPU 조달.
- 멀티 서버 운영 전략 추가(07 §1.1): 사이트 간 DDP 금지·실험 병렬화, 주력(장기 학습)/보조(짧은 실험) 역할 분담, 부트스트랩 스크립트 단일화, 서버 프로필 config 분리, 아티팩트 허브(HF private) 경유, local_lb.csv에 server 컬럼, 최종 추론은 단일 서버 고정

- GPU 렌트 전 준비 완료: pyproject+uv.lock(킷 정합, omegaconf 버그 반영), setup_server.sh(sha256 내장), results/ 정본 CSV 2종, E0 런북([09_E0_런북.md](09_E0_런북.md), exp04). E0는 추론만이라 저가 GPU로 충분 — 96GB 학습 서버 결정은 E3 시점으로 분리
- open.zip 반입 경로 확인: 데이콘 CDN(cfiles.dacon.co.kr)이 **인증 없이 접근 가능**(HEAD 200·크기 일치) → 서버에서 직접 다운로드, setup_server.sh에 반영. scp 불필요 (백업: 집 업링크 35.5Mbps 실측, ~32분)
- **로컬 리더보드 구축 + GT self-test** → [08_로컬리더보드_구축.md](08_로컬리더보드_구축.md): local_eval/ 완성(킷 브리지·2단 홀드아웃·채점기). ★핵심 발견: 킷 action extractor가 train 도메인 다수에서 신뢰 불가(GT 영상 MAE 1.49 > 베이스라인 eval 0.579) → 로컬 Action은 eval 실측을 정본으로 변경. 정지영상의 DINO/Video는 0.03~0.04로 낮음 → Action 지배 재확인

### 리더보드 스냅샷 (7/18, 대회 3일차)
- 1위 0.27184, 2~3위 0.30241(동점), 중위 0.336~0.517, **다수(35위 이하)가 0.51708에 수렴** → 0.517 ≈ 베이스라인 그대로 제출한 점수로 추정 (역산: Action 0.579×0.4=0.232 → DINO+Video 평균 cos dist ≈ 0.48)
- 제출 41팀 / 참가 98명

### 후속 조사 목록 (EDA 신규 발견, 상세는 [05_EDA검증.md](05_EDA검증.md) §5)
- [x] ① action-CFG 조사 완료 → [06_조사_action_cfg.md](06_조사_action_cfg.md): 베이스라인엔 학습된 uncond 경로가 전무(텍스트 uncond는 no-op이었음). E1에서 CFG 봉인, 자체 학습 시 action dropout 0.1 포함 결정
- [ ] ② <16프레임 에피소드 34개: 데이터 로더 필터 구현 시 반영 (전체의 0.31%라 영향 미미하나 크래시 방지 필수)
- [ ] ③ eval wrist_roll +1.5σ 시프트: eval 216개 action 분포를 train과 정밀 대조 → unseen-scene 홀드아웃을 eval 분포에 가깝게 설계할 수 있는지 조사
- [ ] ⑤ eval↔train 장면 겹침 전수 대조: eval 이미지 216장 vs train 11,132 에피소드 첫 프레임 매칭 (저해상도 1차 스크리닝 → 후보 정밀 비교). 겹침 있으면 해당 데이터셋 오버샘플링(합법), 없으면 "eval 100% 미공개 장면" 확정 → ③의 홀드아웃 설계 근거. 정황상 eval은 주최측 자체 수집 추정(camera_key 'top' 잔재, action 분포 시프트)
- [ ] ④ 카메라 키 예외 `liyitenga/so100_bi_giveme5`(s_left): 영상 뷰가 특이한지 확인, 필요 시 학습 제외 후보
