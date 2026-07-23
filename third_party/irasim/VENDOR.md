# VENDORED: IRASim (ByteDance)

- **Upstream**: https://github.com/bytedance/IRASim
- **Commit**: `c72b6dade6fcd65971e0aa8ab49ea39b15108c90` (2025-07-08)
- **Vendored**: 2026-07-21 (W2-M0)
- **License**: Apache-2.0 (원본 [LICENSE](LICENSE) 보존, 코드 파일 헤더 유지)

## 반입 방식
- `git clone --depth 1` 후 `.git/` 제거하여 소스만 반입 (서브모듈 아님 — 오프라인 재현·핀 고정 목적).
- **제거한 것**: `assets/`(README 데모 미디어 3.8M, 코드 미참조 — `grep assets/ main.py util.py models/` 0건 확인). 코드·config·scheduler는 무삭제.
- **무추가/무수정**: upstream 파일은 그대로. 우리 통합 코드는 절대 이 디렉토리에 넣지 않고 `impl/w2/`에만 둔다 (upstream diff = 0 유지).

## 과제 적합성 (반입 근거, exp09/exp12)
- `configs/*/frame_ada.yaml`: `num_frames=16`, `mask_frame_num=1`(1 hist), `extras=3`(frame-level AdaLN 조건) → 과제(1 hist + 15 action → 16f) 동형.
- VAE = SDXL(`pretrained_models/stabilityai/...`), f8 프레임별 → 시간압축 미스매치 없음.
- `attention_mode: 'math'` 지원 → flash-attn 미사용 폴백(V100 sm_70) 확보. `flash_attn`/`xformers`는 `models/irasim.py`에서 try/except (없어도 동작).

## 다음 (M1에서)
- 액션 임베더 EE-traj → SO100 6관절 교체는 **여기 수정 금지**, `impl/w2/` 어댑터로 래핑.
- ckpt(Bridge/Language-Table)는 [impl/w2/download_irasim_ckpts.sh](../../w2/download_irasim_ckpts.sh)로 별도 반입.
