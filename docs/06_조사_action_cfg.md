# 후속 조사 ① — action-CFG 가능 여부 (baseline 코드 정독, 2026-07-18)

> 근거 코드: `openaimodel3d.py` (action 임베딩·dropout), `ddpm3d.py:1294 prepare_batch_for_inference` (uncond 구성), `ddim.py:258` (CFG 분기), `ldwma/datasets/lerobot_so100.py` (caption 처리)

## 결론
**베이스라인 체크포인트에는 학습된 uncond 경로가 하나도 없다 (action뿐 아니라 텍스트·이미지도).** CFG>1은 "학습 안 된 예측과의 대비"라 이론적 근거 없음. 제대로 된 action-CFG는 `action_dropout_prob>0`으로 (파인)튜닝해야 열린다 — **config 한 줄이면 되고 구조 변경은 불필요**.

## 상세 근거
1. **구조는 CFG-ready**: UNet에 `null_action_emb`(학습 가능한 "행동 없음" 임베딩) 슬롯이 이미 있고, forward에서 act=None이면 이걸 사용. 즉 action-CFG용 설계는 되어 있음.
2. **그러나 학습이 안 됨**: `null_action_emb`는 zeros로 초기화되는데, `action_dropout_prob=0`이라 학습 중 한 번도 선택되지 않음 → 기울기 0 → **지금도 zeros 그대로** (모델이 "행동 없음"을 본 적 없음).
3. **텍스트 uncond도 무의미**: 학습 데이터의 caption이 항상 ""(use_language=False)여서, uncond_prob 5%로 caption을 ""로 바꾸는 처리는 **""을 ""로 바꾼 no-op**이었음.
4. **추론 uncond 구성** (`prepare_batch_for_inference`): uc = [빈 텍스트 임베딩 + **검은 이미지** CLIP 임베딩 + 조건프레임 latent(유지) + act 키 없음→null emb]. 검은 이미지 uncond도 학습된 적 없음 → uncond 예측은 이중으로 OOD.
5. CFG 분기(`ddim.py`)는 scale=1.0이면 uncond 자체를 건너뜀 — 현 baseline 설정에서 uc는 만들어지되 사용 안 됨.

## E1/E4 설계 반영
- **E1(무학습)**: action-CFG는 봉인. 개선 수단은 η=0 결정론화·스텝 수·배치 병렬로 한정. (CFG 소폭 스윕은 비용이 싸니 로컬 점수로 한번 확인만 — 기대는 낮음.)
- **베이스라인 트랙에서 CFG를 원하면**: 기존 ckpt에서 `action_dropout_prob≈0.1`로 이어서 파인튜닝 → null_action_emb가 학습되며 CFG 개방. 비용은 GPU 시간뿐, E1 결과가 좋을 때만 투자.
- **주력(Wan2.2) 트랙**: 우리가 action 모듈을 새로 다는 것이므로 **처음부터 action dropout(~0.1)을 넣고 학습** — 추가 비용 없이 CFG 옵션 확보. 파이프라인 S2 설계에 반영할 것.
