# `src/` — 학습·추론·평가 코드

INChall26 학습 폴더의 코드 루트. 엔트리포인트는 `python -m src.cli` (train/generate/score/config).

## 모듈 맵
| 경로 | 역할 | 상태 |
|---|---|---|
| `cli.py` | 통합 엔트리포인트(서브커맨드 디스패치) | 동작 |
| `utils/config.py` | base+exp+server yaml 3계층 딥머지 + CLI override | 동작 |
| `utils/runtime.py` | seed 고정 · logger · device/프로파일 | 동작 |
| `data/dataset.py` | 열거·제외·커리큘럼 + ClipDataset(on-the-fly)·LatentDataset | **이식됨** |
| `data/transforms.py` | 320×512 킷 letterbox · z-score · 통계로더 | **이식됨** |
| `data/build_latent_cache.py` | SDXL VAE 사전인코딩 캐시 생성(pre_encode) | **이식됨**(GPU 실행) |
| `models/backbone.py` | `load_backbone`(irasim_runtime.build 위임) · FT | **이식됨** |
| `models/irasim_runtime.py` | 벤더 래퍼: build·encode_video·training_loss·eval_loss_grid·build_sampler·generate | **이식됨** |
| `models/action_adapter.py` | `adapt_action_seq`(정렬) + `swap_action_embedder`(EE7→SO100 6, zero-init) | **이식됨** |
| `../third_party/irasim/` | 벤더 IRASim 원본(무수정, Apache-2.0, 1.6M) — get_models·diffusion·pipeline·configs | 반입됨 |
| `train/trainer.py` | 학습 루프 · FT선택 · ckpt회전/재개 · det_loss eval | **이식됨** |
| `infer/generate.py` | 결정론 생성(η=0, 16f) · 홀드아웃 조건준비 | **이식됨** |
| `eval/local_score.py` | local_eval 킷 브리지 채점(DINO+Video+Action) 재사용 | **이식됨** |

## 데이터 흐름 (IRASim)
```
data(pre-encode latent) → models(IRASim DiT + SO100 임베더) → train(디퓨전 손실 루프)
                                         └ eval-every → eval/local_score → results/local_lb.csv → best ckpt
best ckpt → infer/generate(PNDM, η=0, 16f, SDXL 디코드) → local_runs/ → eval/local_score(제출 전 검증)
```

## 규약
- **import 경로는 `src.` 절대 임포트** (`python -m src.cli` 기준). 실행은 INChall26/ 를 CWD로.
- stub은 전부 `raise NotImplementedError` + docstring에 구현 골격/근거. 진입 시 docstring부터 확인.
- 새 자체 채점기 만들지 말 것 — 채점은 반드시 `local_eval/` 킷 브리지 재사용(서버 산식 정합).
- **★실행 요건(모델 계층)**: IRASim 학습/생성은 **`diffusers`** 필요(INChall26 기본 `.venv` 미포함 → pyproject 추가) + GPU + IRASim 슬림 ckpt. 데이터 계층(dataset/transforms/build_latent_cache)은 diffusers 없이 동작.
- **벤더 무수정**: `third_party/irasim/`는 원저자 원본(Apache-2.0, 수정 금지). 우리 개조는 전부 `src/`에서 래핑(예: `action_adapter.swap_action_embedder`).
- **이식 원본**: 남은 stub(`train/trainer.py`·`infer/generate.py`)은 다른 repo `../IRASim-SO100-Fine-tuning/impl/w2/`(`train_w2.py`·`gen_eval.py`)에서 이식. 채점은 `local_eval/` 킷 브리지, 큐레이션·정규화 근거는 `docs/17_E2_학습전략_초안.md`.
