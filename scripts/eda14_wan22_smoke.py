"""Wan2.2-TI2V-5B가 이 V100(32GB)에 올라가는지 확인하는 적재 시험.

■ 목적과 범위
E2 단계 후보 모델인 Wan2.2-TI2V-5B가 이 서버 GPU에 적재 가능한지만 본다.
**추론 실행이나 실험 설계는 T 영역**이므로 하지 않는다(48h 계획 재량 항목 단서).
결과는 BOARD에 정보로만 제공한다.

■ 왜 이 방식인가
정식으로 모델을 돌리려면 Wan 코드베이스를 설치해야 하는데, 그러면 uv.lock으로
고정한 환경(룰북 §5 정본)에 의존성 충돌 위험이 생긴다. 대신 가중치 텐서를 직접 읽어
fp16으로 GPU에 올려 **메모리에 들어가는지만** 잰다. 의존성을 건드리지 않는 저위험 방식이다.

■ 무엇을 재는가
  - 확산 모델(diffusion) 가중치를 fp16으로 올렸을 때 점유 메모리
  - 텍스트 인코더(T5)까지 함께 올렸을 때 점유 메모리
  - 각각 32GB 안에 들어가는지, 추론용 여유(활성값 저장 공간)가 남는지

용어: fp16/fp32(숫자를 16비트/32비트로 저장 — fp16은 절반 용량, V100은 bf16 미지원 세대라
      fp16이 정답) · safetensors(가중치 저장 형식) · 활성값(추론 중 생기는 중간 계산 결과).

사용: .venv/bin/python scripts/eda14_wan22_smoke.py [--model-dir /home/jovyan/models/wan22]
산출: results/wan22_smoke.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def gb(x: int) -> float:
    return round(x / 2**30, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/jovyan/models/wan22")
    args = ap.parse_args()
    d = Path(args.model_dir)

    if not torch.cuda.is_available():
        raise SystemExit("GPU 없음")
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    out: dict = {"gpu": props.name, "sm": f"sm_{props.major}{props.minor}",
                 "gpu_total_gb": gb(props.total_memory),
                 "bf16_supported_api": torch.cuda.is_bf16_supported()}
    try:
        out["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        out["bf16_native"] = None
    print(f"{out['gpu']} {out['sm']} {out['gpu_total_gb']}GB", flush=True)
    print(f"  bf16: API {out['bf16_supported_api']} / 네이티브 {out['bf16_native']}", flush=True)

    # ★ 함정: is_bf16_supported()는 에뮬레이션까지 포함해 True를 반환한다.
    #   V100(sm_70)에는 bf16 텐서코어가 없어 실제로는 fp32보다도 느리다. 실측으로 확인한다.
    import time as _t
    out["matmul_tflops"] = {}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)):
        a = torch.randn(4096, 4096, device="cuda:0", dtype=dt)
        b = torch.randn(4096, 4096, device="cuda:0", dtype=dt)
        for _ in range(5):
            a @ b
        torch.cuda.synchronize()
        t0 = _t.time()
        for _ in range(50):
            a @ b
        torch.cuda.synchronize()
        out["matmul_tflops"][name] = round(2 * 4096 ** 3 * 50 / (_t.time() - t0) / 1e12, 1)
        del a, b
    torch.cuda.empty_cache()
    print(f"  행렬곱 실측 TFLOPS: {out['matmul_tflops']}", flush=True)
    torch.cuda.reset_peak_memory_stats()

    from safetensors.torch import load_file  # transformers 의존으로 이미 설치돼 있음

    # --- 1) 확산 모델 ---
    shards = sorted(d.glob("diffusion_pytorch_model-*.safetensors"))
    held = []
    n_params = 0
    for s in shards:
        sd = load_file(str(s))
        for k, v in sd.items():
            n_params += v.numel()
            held.append(v.to("cuda:0", dtype=torch.float16))
        del sd
    torch.cuda.synchronize()
    out["diffusion_params_b"] = round(n_params / 1e9, 2)
    out["diffusion_fp16_gb"] = gb(torch.cuda.memory_allocated())
    print(f"확산 모델: {out['diffusion_params_b']}B 파라미터, "
          f"fp16 적재 {out['diffusion_fp16_gb']}GB", flush=True)

    # --- 2) 텍스트 인코더(T5) 추가 ---
    t5 = d / "models_t5_umt5-xxl-enc-bf16.pth"
    if t5.exists():
        sd = torch.load(t5, map_location="cpu", weights_only=True)
        t5_params = 0
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                t5_params += v.numel()
                held.append(v.to("cuda:0", dtype=torch.float16))
        del sd
        torch.cuda.synchronize()
        out["t5_params_b"] = round(t5_params / 1e9, 2)
        out["diffusion_plus_t5_fp16_gb"] = gb(torch.cuda.memory_allocated())
        print(f"+ T5 인코더: {out['t5_params_b']}B, 합계 {out['diffusion_plus_t5_fp16_gb']}GB",
              flush=True)

    out["peak_gb"] = gb(torch.cuda.max_memory_allocated())
    out["headroom_gb"] = round(out["gpu_total_gb"] - out["peak_gb"], 2)
    out["fits"] = out["headroom_gb"] > 0
    # 추론에는 활성값 저장 공간이 추가로 필요하다. 경험적으로 확산 영상 생성은
    # 가중치 외에 수 GB를 더 쓰므로, 여유가 그보다 작으면 실행이 어렵다고 본다.
    out["verdict"] = (
        "적재 가능 — 추론 여유도 충분" if out["headroom_gb"] >= 6 else
        "적재는 되나 추론 여유 빠듯 — 텍스트 인코더 오프로딩·해상도 축소 필요" if out["fits"] else
        "적재 불가 — 32GB 초과"
    )
    print(f"\n최대 점유 {out['peak_gb']}GB / 여유 {out['headroom_gb']}GB → {out['verdict']}")

    (ROOT / "results" / "wan22_smoke.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print("→ results/wan22_smoke.json")

    del held
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
