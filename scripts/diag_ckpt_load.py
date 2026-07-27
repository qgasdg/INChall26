"""체크포인트가 실제로 실렸는지 검사 — 키 대조 + 가중치 통계 (GPU 불필요).

배경: `diag_train_repro.py --ckpt ckpts/e3r4_s14000_best.pt` 결과가 16개 샘플 전부
      **출력 선명도 1072로 동일**, PSNR 5.8dB로 바닥이었다. 입력이 뭐든 같은 걸 뱉는다는
      뜻이고, 이는 가중치가 안 실려 **난수 초기 상태**로 생성했을 때의 전형적 증상이다.

검사 2종:
  1) 키 대조 — ckpt에 든 텐서 이름이 모델이 기대하는 이름과 맞는가. 접두사(`module.` 등)만
     달라도 전부 버려진다. 매칭률이 낮으면 그게 원인.
  2) 가중치 통계 — 실제로 로드한 뒤 층별 평균·표준편차가 사전학습 ckpt와 얼마나 다른가.
     학습된 가중치는 층마다 값이 제각각인데, 난수 초기화는 표준편차가 층 크기 공식대로
     매끈하게 나온다. 기준 ckpt와 나란히 찍어 눈으로 가른다.

용어: 키(가중치 이름표) · state_dict(이름표→가중치 표) · 접두사(이름 앞에 붙는 꼬리표).

사용: /opt/conda/envs/torch260/bin/python scripts/diag_ckpt_load.py \
        --ckpt ckpts/e3r4_s14000_best.pt --ref ckpts/bridge_frame_ada_0300000.pt
산출: 표준출력 + results/diag_ckpt_load.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _unwrap(raw):
    """torch.load 결과에서 실제 state_dict를 꺼낸다. 어느 칸에서 나왔는지도 함께 반환."""
    import torch

    if not isinstance(raw, dict):
        return raw, "(dict 아님)"
    if all(isinstance(v, torch.Tensor) for v in raw.values()) and raw:
        return raw, "(최상위가 곧 state_dict)"
    for k in ("ema", "model", "state_dict", "module", "net", "weights"):
        if k in raw and isinstance(raw[k], dict):
            return raw[k], k
    return raw, "(못 찾음 — 최상위 그대로)"


def inspect(path: str, model_keys: set[str]) -> dict:
    import numpy as np
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    top = sorted(raw.keys())[:12] if isinstance(raw, dict) else []
    sd, where = _unwrap(raw)
    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    keys = set(tensors)

    # 접두사만 다른 경우를 잡아낸다 (module. / model. / _orig_mod. 등)
    stripped = {}
    for pref in ("module.", "model.", "_orig_mod.", "net.", "diffusion_model."):
        cand = {k[len(pref):] for k in keys if k.startswith(pref)}
        if cand:
            stripped[pref] = len(cand & model_keys)

    matched = keys & model_keys
    out = {
        "path": path, "top_level_keys": top, "state_dict_from": where,
        "n_tensors": len(tensors), "n_model_keys": len(model_keys),
        "n_matched": len(matched), "match_pct": 100.0 * len(matched) / max(len(model_keys), 1),
        "missing_in_ckpt": sorted(model_keys - keys)[:8],
        "unexpected_in_ckpt": sorted(keys - model_keys)[:8],
        "prefix_strip_would_match": stripped,
    }
    if isinstance(raw, dict):
        for k in ("step", "global_step", "epoch", "iter"):
            if k in raw and not isinstance(raw[k], dict):
                out[k] = int(raw[k]) if hasattr(raw[k], "__int__") else str(raw[k])

    # 매칭된 층 중 큰 것 몇 개의 통계 — 학습된 값인지 난수인지 눈으로 가르기
    stats = []
    for k in sorted(matched, key=lambda k: -tensors[k].numel())[:5]:
        t = tensors[k].float().numpy()
        stats.append({"key": k, "shape": list(t.shape),
                      "mean": float(np.mean(t)), "std": float(np.std(t)),
                      "absmax": float(np.abs(t).max())})
    out["top_layer_stats"] = stats

    print("\n--- %s" % path)
    print("  최상위 칸: %s" % (", ".join(top) if top else "(없음)"))
    print("  state_dict 위치: %s · 텐서 %d개" % (where, len(tensors)))
    print("  키 매칭 %d/%d (%.1f%%)" % (len(matched), len(model_keys), out["match_pct"]))
    if out["match_pct"] < 90:
        print("  ★ 매칭률이 낮다 — 이름이 안 맞아 가중치가 버려졌을 가능성")
        if out["unexpected_in_ckpt"]:
            print("  ckpt에만 있는 이름 예: %s" % ", ".join(out["unexpected_in_ckpt"][:4]))
        if out["missing_in_ckpt"]:
            print("  모델이 원하는데 없는 이름 예: %s" % ", ".join(out["missing_in_ckpt"][:4]))
        for pref, n in stripped.items():
            print("  '%s' 접두사를 떼면 %d개 매칭 → 접두사 문제" % (pref, n))
    for s in stats:
        print("    %-40s std %.4f  |최대| %.3f" % (s["key"][:40], s["std"], s["absmax"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="검사할 체크포인트")
    ap.add_argument("--ref", default="ckpts/bridge_frame_ada_0300000.pt",
                    help="비교 기준(정상 동작 확인된 사전학습 ckpt)")
    args = ap.parse_args()

    from src.models.irasim_runtime import _add_vendor_path, build_vendor_cfg

    _add_vendor_path()
    from models import get_models

    model = get_models(build_vendor_cfg(gen_hw=(320, 512), num_frames=16))
    model_keys = set(model.state_dict())
    print("모델이 기대하는 텐서 %d개" % len(model_keys))

    res = {"target": inspect(args.ckpt, model_keys)}
    if args.ref and Path(args.ref).exists():
        res["ref"] = inspect(args.ref, model_keys)
        print("\n=== 결론 ===")
        t, r = res["target"]["match_pct"], res["ref"]["match_pct"]
        print("  대상 %.1f%% vs 기준 %.1f%%" % (t, r))
        print("  %s" % ("→ 적재 문제 확정. 기준은 붙는데 대상은 안 붙는다 — 저장 형식이 다르다"
                        if r - t > 10 else
                        "→ 적재는 정상. 생성 이상은 가중치 자체(학습 발산 등)를 의심할 것"))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "diag_ckpt_load.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ results/diag_ckpt_load.json")


if __name__ == "__main__":
    main()
