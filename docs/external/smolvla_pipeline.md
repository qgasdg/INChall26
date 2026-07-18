# SmolVLA 전체 파이프라인 정리 — 동일 공개 데이터(SO-100 커뮤니티 데이터셋) 활용 사례

> 논문: SmolVLA: A vision-language-action model for affordable and efficient robotics (Hugging Face 외, arXiv:2506.01844, 2025-06)
> 링크: https://arxiv.org/abs/2506.01844 · 블로그: https://huggingface.co/blog/smolvla · 모델: https://huggingface.co/lerobot/smolvla_base · 코드: https://github.com/huggingface/lerobot
> 작성일: 2026-07-16. 본 문서의 수치는 논문 원문(v1)에서 직접 확인한 값이며, 논문에 없는 구현 세부는 [구현]으로 표기.

## 0. 대회와의 관계 (읽는 이유)

SmolVLA는 우리 대회 train 데이터와 **같은 SO-100 LeRobot 커뮤니티 데이터셋**으로 사전학습된 대표 연구다 (부록 A.1 목록과 대회 128개 대조 → **최소 101개(79%) 일치 확인**). 단, 방향이 다르다: SmolVLA는 **VLA(영상+언어 → 행동 예측 정책)**, 우리 대회는 **월드 모델(영상+행동 → 미래 영상 생성)**. 즉 입출력이 반대다. 그래서 모델 구조 자체보다 **① 커뮤니티 데이터 전처리·큐레이션 노하우, ② 학습 레시피(효율화 기법), ③ 평가 설계**가 차용 포인트다. 참고로 대회 평가용 action extractor(영상→행동 회귀)는 개념상 SmolVLA와 같은 "영상→행동" 방향이다.

---

## 1. 데이터 수집·큐레이션

| 항목 | 내용 |
|---|---|
| 출처 | Hugging Face Hub의 커뮤니티 업로드 LeRobot 데이터셋 |
| 규모 | **481개 데이터셋, 22.9K 에피소드, 10.6M 프레임** (Table 1) — 기존 대형 VLA 대비 1자릿수 이상 작음 |
| 선별 기준 | **임베디먼트 타입, 에피소드 수, 전반적 데이터 품질, 프레임 커버리지**로 필터링 (논문에 구체 임계값은 미공개) |
| 특성 | 표준 프로토콜 없는 실환경 데이터: 다양한 로봇 형태·제어 방식·카메라 시점·노이즈 낀 시연 포함 — 이를 "가치 있는 사전학습 데이터"로 봄 |

### 1-1. 태스크 주석 재생성 (VLM 자동 주석)

문제: 커뮤니티 데이터의 태스크 설명이 부실 (`task_desc` 같은 플레이스홀더, "Hold"/"Up" 같은 모호한 명령, 아예 없는 경우).
해결: **Qwen2.5-VL-3B-Instruct**로 대표 프레임 + 기존(노이즈) 설명을 주고 짧은 행동 중심 문장을 자동 생성. 논문 부록의 실제 프롬프트:

> "Here is a current task description: {current_task}. Generate a very short, clear, and complete one-sentence describing the action performed by the robot arm (max 30 characters). Do not include unnecessary words. Be concise. Here is some examples: Pick up the cube and place it in the box, open the drawer and so on. Start directly with an action verb like 'Pick', 'Place', 'Open', etc. Similar to the provided examples, what is the main action done by the robot arm?"

→ **대회 적용**: 대회 데이터도 동일한 품질 문제 확인됨("test", 빈 문자열 등). 언어 조건을 쓰는 모델을 실험한다면 같은 방식의 로컬 VLM 재주석이 유효 (공개 가중치 + 허용 라이선스 확인 시 규정상 가능. 외부 '데이터'가 아니라 제공된 데이터에 대한 주석 생성이지만, 규정 해석이 애매하면 주최측 문의 권장).

### 1-2. 카메라 시점 표준화

문제: `images.laptop` 같은 이름이 데이터셋마다 top/side/wrist 중 무엇인지 제각각 → 사전학습에 유해함을 확인.
해결: 각 카메라를 **수동으로** top/wrist/side 유형에 매핑해 OBS_IMAGE_1(top), OBS_IMAGE_2(wrist), OBS_IMAGE_3(side)로 리네이밍. 초과 뷰는 학습에서 제외. **일관된 카메라 순서가 이 데이터 규모에서는 성능에 도움**이라고 보고.
→ **대회 적용**: 대회 데이터는 이미 카메라 1개로 축소되어 직접 해당 없음. 단 "시점 이질성이 성능에 유해"하다는 발견은 유효 — 시점 클러스터링(탑뷰/사이드뷰 구분) 후 조건 부여 또는 필터링 실험 가치 있음.

### 1-3. 기타 전처리

- 이미지: **512×512로 리사이즈** (VLM 입력 규격 일치 목적).
- 고정 시퀀스 길이·배치 크기 유지(torch.compile 호환), 배치에 안 맞는 에피소드 잔여 프레임은 버림.
- action/state 정규화: 논문 본문에 명시 없음. [구현] LeRobot 프레임워크는 데이터셋 통계 기반 정규화(mean/std)를 표준으로 사용 — 대회 제공 `so100_action_statistics.json`과 같은 방식.

## 2. 모델링

구성: **컴팩트 사전학습 VLM(지각) + Flow Matching 액션 전문가(행동 생성)**, 총 **450M** 파라미터(액션 전문가 ~100M). 변형: 0.24B / 0.45B(main) / 2.25B.

### 2-1. VLM 백본 (SmolVLM-2)

- SigLIP 비전 인코더 + SmolLM2 언어 디코더.
- 입력: 멀티 카메라 RGB + 자연어 지시 + 센서모터 state (state는 선형 프로젝션으로 VLM 차원에 매핑해 **VLM에 prefix 토큰으로 입력** — ablation에서 액션 전문가에 직접 넣는 것보다 유의하게 좋음).
- **레이어 스키핑**: LLM의 앞 **N = L/2 레이어만** 사용(main 모델은 첫 16층) → 연산 절반, 성능 유지.
- **비주얼 토큰 최소화**: 타일링 없이 글로벌 이미지 + pixel shuffle로 **프레임당 64토큰**.

### 2-2. 액션 전문가 (Flow Matching Transformer)

- 출력: **n = 50 스텝 액션 청크** (연속값, 회귀 아님).
- **Cross-Attention(VLM feature를 K/V로) ↔ Causal Self-Attention(청크 내 과거 액션만 참조) 교차 배치** — ablation: 교차 85.5% > CA만 79.0% > SA만 74.5%, causal > bidirectional(74.5 vs 67.5, 미래 액션 누출 방지 효과).
- hidden dim은 VLM의 **0.75배**로 축소.
- 학습 목적함수: **Flow Matching** — L1 회귀 대비 유의한 우위 (LIBERO 80.25% vs 75.25%).

→ **대회 시사점**: (역방향이지만) "행동↔영상 결합 모듈은 cross-attn + 인과 구조가 유리", "연속값 생성엔 flow/diffusion류가 회귀보다 강함"은 월드 모델의 action 주입 설계에도 참고 가능. 대회 베이스라인(DynamiCrafter UNet의 action_conditioned 주입)과 비교 실험 후보.

## 3. 학습

| 항목 | 값 |
|---|---|
| 프레임워크 | LeRobot (PyTorch) |
| 사전학습 | **200K step, global batch 256**, 커뮤니티 데이터셋 전체 |
| LR 스케줄 | warmup 100 step → cosine, **1e-4 → 2.5e-6** |
| 옵티마이저 | AdamW (β1=0.9, β2=0.95) |
| 정밀도/최적화 | bfloat16, torch.compile(JIT), HF accelerate (멀티GPU) |
| 학습 대상 | **VLM 동결, 액션 전문가만 학습** |
| 자원 | 사전학습은 GPU 4장(대배치용), 단일 GPU로도 가능한 크기. 프로젝트 전체 ~**30K GPU시간** |
| 파인튜닝 | 시뮬 100K step(batch 64), 실기 200K step — "훨씬 적은 step으로도 큰 손실 없음" 언급 |

→ **대회 적용**: 학습 4일 × RTX PRO 6000 1장 제한과 비교하면, "동결 백본 + 소형 전문가 모듈만 학습" 전략이 시간 예산에 잘 맞음 (베이스라인도 실제로 backbone 동결 + UNet 11M 학습 구조). bf16 + torch.compile + 고정 시퀀스 길이는 그대로 차용 가능.

## 4. 추론 (비동기 스택)

- 문제: 청크 단위 정책은 청크 소진 후 다음 예측까지 로봇이 대기(동기) 또는 매 스텝 추론(고비용).
- 해결: **RobotClient/PolicyServer 분리**, 액션 큐 잔량이 임계값 **g(=0.7)** 아래로 내려가면 새 관측을 보내 **실행 중에 다음 청크를 미리 예측**, 겹치는 구간은 병합. 유사 관측 필터로 중복 처리 방지.
- 결과: 성공률 동등, **태스크 완료 ~30% 단축**(9.7s vs 13.75s), 고정 60초 내 사이클 19회 vs 9회.
- flow matching 추론 스텝 10 고정. 실기 평가는 동기 모드(청크 완전 소진 후 재관측), 시뮬은 매 스텝 재관측.

→ **대회 시사점**: 실시간성은 대회와 무관하나, **추론 1시간 제한** 대응으로 배치 병렬 생성·스텝 수 절감(예: DDIM 50→적정선) 튜닝이 같은 성격의 엔지니어링.

## 5. 평가·실험 설계

### 5-1. 벤치마크와 결과

시뮬레이션 (멀티태스크 학습, 태스크당 10 trial, 성공=1/실패=0):

| 벤치마크 | 데이터 | SmolVLA 0.45B | 주요 비교 |
|---|---|---|---|
| LIBERO (40태스크: Spatial/Object/Goal/Long) | 1,693 에피소드 | **87.3%** (90/96/92/71) | π0 3.3B(로봇 사전학습) 86.0, OpenVLA 7B 76.5, Octo 75.1, π0(VLM만) 71.8 |
| Meta-World MT50 (easy~very hard) | 2,500 에피소드(50/task) | **57.3%** | π0(VLM만) 50.5, π0(사전학습) 47.9, TinyVLA 31.6 |
| 스케일 | — | 2.25B: LIBERO 88.75 / MW 68.24 | 0.24B: 82.75 / 56.95 |

실기 (SO-100 3태스크 + SO-101 1태스크; 각 데이터셋 = 시작위치 5종 × 10궤적 = 50 시연; 부분점수 0.5+0.5):

| 설정 | 결과 |
|---|---|
| SO-100 (Pick-Place / Stacking / Sorting, 멀티태스크) | SmolVLA **78.3%** vs π0 61.7% vs ACT(단일태스크) 48.3% |
| SO-101 Pick-Place-Lego (단일태스크, 사전학습에 SO-101 없음) | ID **90%** / OOD **50%** vs ACT 70/40 |
| 사전학습 효과 | 커뮤니티 데이터 사전학습으로 51.7 → **78.3** (+26.6pt), 멀티태스크 파인튜닝 추가 이득 |

평가 데이터도 공개: `lerobot/svla_so100_pickplace`, `svla_so100_stacking`, `svla_so100_sorting`, `svla_so101_pickplace` (HF).

### 5-2. Ablation 요약 (LIBERO)

- 어텐션: CA+SA 교차 85.5 > CA 79.0 > SA 74.5 / causal > bidirectional
- VLM 레이어: 절반만 사용이 소형 VLM 통째보다 나음 (연산-성능 균형)
- 전문가 폭: 0.75× 최적 / 목적함수: FM 80.25 > L1 75.25
- state 입력 위치: VLM 프리픽스 ≫ 전문가 직접 입력
- 청크 크기 10~50 최적, 관측 갱신 주기 짧을수록 SR↑ (속도와 트레이드오프)

### 5-3. 실험 방법론에서 배울 점 (대회 실험 설계 반영)

1. **부분점수 평가**: 실기 태스크를 하위 단계로 쪼개 0.5+0.5 채점 → 우리도 로컬 평가에서 성분별(DINO/Video/Action) + 관절별(특히 gripper) 분해 리포트를 표준화할 것.
2. **단일 변수 ablation 표**: 어텐션/레이어 수/목적함수/청크 크기 각각 분리 실험 → experiment log 형식으로 동일하게 운용.
3. **사전학습 유/무 비교를 별도 표로** → 우리도 "베이스라인 ckpt에서 계속 학습 vs 백본에서 재학습" 비교를 초기 실험으로.
4. **평가 프로토콜 고정**(trial 수, 시작 위치) 후 모델만 교체 → 로컬 홀드아웃 세트·시드·생성 설정 고정이 선행 조건.

## 6. 한계 (논문 자인 + 대회 관점)

- 논문: 데이터 다양성 한계, 장기 태스크 확장성, VLM 백본 의존.
- 대회 관점: SmolVLA는 정책(행동 예측) 논문이라 **영상 생성 품질에 대한 지표·기법은 없음** — 월드 모델 기법은 Reference Paper 폴더의 계열(DynamiCrafter, IRASim, iVideoGPT, Cosmos, V-JEPA2 등)에서 가져와야 함. SmolVLA에서 가져올 것은 데이터 취급과 실험 방법론이 핵심.

---

### 출처

- 논문 원문: https://arxiv.org/abs/2506.01844 (본 문서 수치는 원문 p.1–11, 부록 A.1에서 확인)
- 블로그: https://huggingface.co/blog/smolvla · 모델 카드: https://huggingface.co/lerobot/smolvla_base
- LeRobot: https://github.com/huggingface/lerobot
- 커뮤니티 데이터셋 배경: https://huggingface.co/blog/lerobot-datasets
