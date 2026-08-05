# DreamZero 노선

IRASim 손절 후 채택한 경로. **왜 이 모델인가**와 **무엇이 아직 안 풀렸나**만 적는다.
확인성 수치 나열은 하지 않는다.

## 무엇을 쓰는가

3층 구조다. 아래가 없으면 위가 동작하지 않는다.

```
Wan2.1-I2V-14B-480P        Apache-2.0, 알리바바. 이미지+텍스트 → 비디오 확산. 액션 개념 없음
  └ DreamZero              Apache-2.0, NVIDIA GEAR. 액션 토큰 스트림 + blockwise causal
    │                      attention + flow matching 액션 헤드. DROID(Franka)로 학습
    └ DreamZero-SO101 LoRA Apache-2.0, Vizuara. rank-4 LoRA 217MB. SO-101 715 에피소드
```

라이선스 사슬 전체가 Apache-2.0이라 대회 규정("공식 공개 가중치 + MIT/Apache/CC BY/CC BY-NC")을
충족한다. LoRA만 허용돼도 베이스가 막히면 못 쓰므로 사슬 전체를 확인해야 한다.

## 왜 IRASim이 아니라 이것인가

IRASim 실패의 핵심은 **액션 표현의 의미 불일치**였다. 사전학습이 7차원 상대 end-effector
이동으로 굳어 있는데 대회는 6차원 절대 관절값이라, 어댑터를 갈아끼워도 백본이 다른 언어를
듣는 상태였다. 스텝을 3배(E4) 밟아도 학습 범위를 3배(E5) 넓혀도 정체가 안 풀린 이유다.

DreamZero-SO101은 그 지점을 정면으로 통과한다.

| | IRASim | DreamZero-SO101 |
|---|---|---|
| 액션 차원 | 7 (상대 EE) → 6 억지 교체 | **6** |
| 액션 축 | WidowX end-effector | **shoulder_pan·shoulder_lift·elbow_flex·wrist_flex·wrist_roll·gripper** |
| 로봇 | WidowX | **SO-101** (우리 SO-100과 같은 계열) |

차원만 같은 게 아니라 **축의 의미와 로봇 계열이 같다**. IRASim에서 우리를 무너뜨린
"차원 일치 ≠ 의미 일치" 함정이 여기엔 없다.

## 조건화 방향 (코드로 확인함)

`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` 의 blockwise
causal 마스크:

- **이미지 블록 → 현재 액션 블록을 attend한다** (우리가 필요한 방향)
- 액션 블록도 과거·현재 이미지를 attend한다 (양방향이다 — "액션은 비디오를 못 본다"는 설명은 틀렸다)
- 첫 이미지는 조건 전용 (우리 과제의 조건 프레임과 같은 역할)

즉 **액션을 조건으로 주고 비디오를 받는 것이 설계 그 자체**다. 별도 주입 인터페이스가 필요 없다.

## 하드웨어 제약 (RTX 5090 32GB)

| | 판정 |
|---|---|
| 추론 | 가능성 있음. bf16 가중치 28GB라 텍스트 인코더(umt5-xxl)를 CPU로 내리거나 fp8 필요 |
| LoRA 학습 | 매우 빠듯. gradient checkpointing + 8bit 옵티마이저 + 오프로드를 다 써야 함 |

5090은 Blackwell이라 **bf16·fp8·flash-attention이 전부 된다**. V100에서 발목을 잡던 제약
(bf16 비네이티브, flash-attn2 불가)이 사라진다. 남은 병목은 VRAM 용량뿐이다.

**백본을 Wan2.2-TI2V-5B로 낮추면 VRAM은 넉넉해지지만 SO-101 LoRA를 못 쓴다**
(hidden dim 5120 vs 3072). DreamZero 사전학습 체크포인트도 5B용은 없어서 처음부터
학습해야 한다 — 이 노선의 이점을 대부분 포기하는 선택이라 최후 수단으로 둔다.

## 단계

**1단계 — 학습 없이 제로샷 생성 (현재)**
SO-101 LoRA를 그대로 얹어 eval216을 생성한다. SO-100/SO-101이 사실상 같은 팔이라 통할
가능성이 있고, 통하면 그 자체로 제출 가능한 결과가 나온다. 학습 여부는 이 결과를 보고 정한다.

**2단계 — SO-100 적응 학습 (조건부)**
1단계 결과가 부족하면 대회 데이터로 추가 LoRA 학습. LeRobot v2 → GEAR 변환 경로가
`docs/DATASET_TO_GEAR_AND_TRAIN.md`에 문서화돼 있고, 우리 데이터가 이미 LeRobot v2다.

## 아직 안 풀린 것

- **액션 조건 비디오 생성 진입점이 저장소에 없다.** `scripts/inference/`엔 TensorRT 빌더뿐이고
  나머지는 로봇 제어 루프다. 모델 구조는 지원하므로 **우리가 생성 스크립트를 작성해야 한다.**
  이게 이 노선의 핵심 산출물이다.
- **추론 예산 실측 미완.** 216샘플 1시간 = 16.7초/샘플. 순정 Wan2.1 확산으로는 14시간이라
  불가능하고, DreamZero의 인과 청크 구조라야 승산이 있다. 실측 전엔 확정 못 한다.
- **해상도 불일치.** DreamZero 기본은 320×176/33프레임, 대회는 512×320/16프레임.
  사전학습 작동점을 벗어나므로 원 해상도 생성 후 업스케일이 나은지 재학습으로 맞출지 실험 필요.
- **액션 표현.** DreamZero DROID 설정은 `relative_action: true`(관절 상대값)인데 대회는 절대값이다.
  상대값은 데이터셋별 관절 원점 차이에 불변이라는 이점이 있어 오히려 유리할 수 있다 —
  어느 쪽으로 넣을지는 1단계에서 양쪽 다 재본다.

## 파일

- `setup_5090.sh` — 환경·가중치 구축
