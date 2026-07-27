"""해상도 A/B — pos_embed 일치(256×320) vs 현행(320×512) 제로샷 뭉개짐 직접 비교.

배경: `diag_blur.py [B]`에서 ckpt의 위치 임베딩이 **320토큰(=256×320)**인데 현행 생성은
      **640토큰(=320×512)**이라 불일치 → 로드 시 드롭(모델 자체값 사용)됨을 확인했다.
      벤더 config(`configs/train|evaluation/bridge/frame_ada.yaml`)도 `video_size: [256,320]`.
      "이 드롭이 뭉개짐의 원인인가"를 **같은 ckpt·같은 샘플·같은 시드로 해상도만 바꿔** 가른다.

지표: 선명도 유지율 = 라플라시안분산(마지막 프레임) / 라플라시안분산(첫=조건 프레임).
      100%에 가까울수록 시간이 지나도 안 뭉개짐. (정적 영상은 정의상 100%)

읽는 법:
  - 256×320이 320×512보다 유지율이 뚜렷이 높다  → 해상도/pos_embed 불일치가 원인 (설정 변경으로 개선 가능)
  - 둘이 비슷하다                                → 원인은 다른 곳(학습 부족·샘플링 설정 등)

용어: pos_embed(모델이 "이 픽셀이 화면 어디쯤인지" 아는 표) · 라플라시안 분산(선명도 수치).

사용(GPU): /opt/conda/envs/torch260/bin/python scripts/diag_res_ab.py --n 3
산출: 표준출력 + results/diag_res_ab.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NATIVE_HW = (256, 320)   # IRASim bridge 사전학습 해상도 = ckpt pos_embed 320토큰
CURRENT_HW = (320, 512)  # 현행 canvas_aligned


def _gray(img: np.ndarray) -> np.ndarray:
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def _lap_var(g: np.ndarray) -> float:
    d = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(d.var())


def _fit_pad(frames: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """(T,H,W,3) uint8 → 종횡비 유지 축소 + 중앙 zero-pad → (T,oh,ow,3). 킷 letterbox와 같은 방식."""
    from src.data.transforms import _interp

    t, h, w = frames.shape[:3]
    oh, ow = out_hw
    s = min(oh / h, ow / w)
    ch, cw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    content = _interp(frames, (ch, cw))
    out = np.zeros((t, oh, ow, 3), dtype=np.uint8)
    y0, x0 = (oh - ch) // 2, (ow - cw) // 2
    out[:, y0:y0 + ch, x0:x0 + cw] = content
    return out


def _samples(cfg: dict, n: int) -> list[dict]:
    """홀드아웃 unseen 앞 n개 — 조건 프레임(원본) + 정규화·정렬된 액션 15스텝."""
    from local_eval import episode_io

    from src.data import transforms as T
    from src.models.action_adapter import adapt_action_seq

    hp = ROOT / "local_eval" / "holdout_v2.json"
    picked = [s for s in json.loads(hp.read_text(encoding="utf-8"))["samples"]
              if str(s.get("tier", "")).startswith("unseen")][:n]
    mean, std = T.load_action_stats()
    seq = int(cfg.get("task", {}).get("frames", 16))
    align = cfg.get("data", {}).get("align_mode", "shifted")
    out = []
    for s in picked:
        ds, ep, st = s["dataset"], int(s["episode_index"]), int(s.get("start", 0))
        f0 = episode_io.read_frames(ds, ep, st, 1)                       # (1,H,W,3) uint8
        acts = T.normalize_action(episode_io.read_actions(ds, ep, st, seq), mean, std)
        out.append({"id": s["sample_id"], "f0": f0,
                    "actions15": np.ascontiguousarray(adapt_action_seq(acts, seq, align))})
    return out


def run_res(cfg: dict, ckpt: str, samples: list[dict], gen_hw: tuple[int, int], steps: int) -> dict:
    """주어진 생성 해상도로 제로샷 생성 → 선명도 유지율. pos_embed 로드 여부도 로그로 남는다."""
    import torch

    from src.models.irasim_runtime import build, encode_video
    from src.models.irasim_runtime import generate as rt_generate

    icfg = copy.deepcopy(cfg)
    icfg.setdefault("task", {})["gen_hw"] = list(gen_hw)
    log = logging.getLogger("res%dx%d" % gen_hw)
    print("\n=== 생성 해상도 %dx%d (토큰 %d) ===" % (gen_hw[0], gen_hw[1], gen_hw[0] * gen_hw[1] // 256), flush=True)
    rt = build(icfg, slim_ckpt=ckpt, logger=log)

    rows = []
    for s in samples:
        cond = _fit_pad(s["f0"], gen_hw)                                  # (1,gh,gw,3)
        x = torch.from_numpy(cond).float().div(127.5).sub(1.0).permute(0, 3, 1, 2)[None]  # (1,1,3,gh,gw)
        mask_x = encode_video(rt, x)
        a15 = torch.from_numpy(s["actions15"]).float()[None]
        pred = rt_generate(rt, mask_x, a15, steps=steps, eta=0.0, method="PNDM",
                           seed=0, gen_hw=gen_hw)                          # (1,16,gh,gw,3) uint8
        v = np.asarray(pred[0])
        s0, s1 = _lap_var(_gray(v[0])), _lap_var(_gray(v[-1]))
        keep = 100.0 * s1 / max(s0, 1e-9)
        rows.append({"sample": s["id"], "sharp_first": s0, "sharp_last": s1, "keep_pct": keep})
        print("  %-46s 선명도 %7.0f → %7.0f (유지 %5.1f%%)" % (s["id"][:46], s0, s1, keep), flush=True)
        del mask_x, pred
        torch.cuda.empty_cache()

    del rt
    torch.cuda.empty_cache()
    mean_keep = float(np.mean([r["keep_pct"] for r in rows]))
    print("  → 평균 유지율 %.1f%%" % mean_keep, flush=True)
    return {"gen_hw": list(gen_hw), "tokens": gen_hw[0] * gen_hw[1] // 256,
            "rows": rows, "mean_keep_pct": mean_keep}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="    %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="비교할 홀드아웃 샘플 수")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ckpt", default="ckpts/bridge_frame_ada_0300000.pt")
    args = ap.parse_args()

    from src.utils.config import load_config

    cfg = load_config()
    cfg.setdefault("train", {})["precision"] = "fp16"      # V100
    cfg.setdefault("data", {})["pre_encode"] = False
    samples = _samples(cfg, args.n)
    print("샘플 %d개: %s" % (len(samples), ", ".join(s["id"][:28] for s in samples)))

    res = {"native": run_res(cfg, args.ckpt, samples, NATIVE_HW, args.steps),
           "current": run_res(cfg, args.ckpt, samples, CURRENT_HW, args.steps)}
    d = res["native"]["mean_keep_pct"] - res["current"]["mean_keep_pct"]
    res["delta_pp"] = d
    print("\n=== 결론 ===")
    print("  256x320(pos_embed 일치) %.1f%%  vs  320x512(드롭) %.1f%%  → 차이 %+.1f%%p"
          % (res["native"]["mean_keep_pct"], res["current"]["mean_keep_pct"], d))
    print("  %s" % ("해상도/pos_embed 불일치가 뭉개짐에 유의미하게 기여" if d > 10 else
                    "해상도만으로는 설명 안 됨 — 다른 원인(학습 부족·샘플링 설정) 우선 확인"))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "diag_res_ab.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ results/diag_res_ab.json")


if __name__ == "__main__":
    main()
