"""생성 영상 뭉개짐(흐림) 원인 진단 — 비용 낮은 순서로 A/B/C 3종.

배경: IRASim 생성물은 프레임이 갈수록 선명도가 떨어진다(측정: 마지막/첫 라플라시안 분산
      비율이 제로샷 19.8% · 파인튜닝 41.7%, 정적은 정의상 100%). 원인이 **모델(학습)**인지
      **VAE/해상도 상한**인지 갈라야 이후 진단이 정해진다.

  A. VAE 상한 — GT 프레임을 encode→decode만 한다. 이게 이미 흐리면 모델 문제가 아니다.
     gen 해상도(320x512)와 native(원본)를 같이 재서 해상도 기여도 분리.
  B. 해상도·pos_embed — ckpt의 위치 임베딩 토큰 수로 원 학습 해상도를 역산하고 현 설정과 대조.
     (학습 로그에 `drop(shape·고정sincos): pos_embed ckpt(1,320,1152) vs model(1,640,1152)` 관측됨)
  C. 노이즈 스케줄 — terminal SNR(마지막 스텝의 신호 잔량). 0이 아니면 저대비·뿌연 출력 경향.

용어: VAE(영상을 압축했다 되돌리는 부품) · 라플라시안 분산(선명도 수치, 높을수록 또렷)
      · PSNR(원본과 얼마나 같은지, 높을수록 같음) · terminal SNR(마지막 확산 스텝의 신호 잔량).

사용(GPU 서버):
  .venv-irasim/bin/python scripts/diag_blur.py --n 4
  # 또는 conda run -n torch260 python scripts/diag_blur.py --n 4
산출: 표준출력 + results/diag_blur.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _gray(img: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 → 밝기 (H,W) float32."""
    return (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).astype(np.float32)


def _lap_var(g: np.ndarray) -> float:
    """라플라시안 분산 = 선명도. 흐릴수록 작다."""
    d = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(d.var())


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def _holdout_frames(n: int) -> list[tuple[str, np.ndarray]]:
    """홀드아웃 unseen 앞 n개의 조건 프레임(원본 해상도)."""
    from local_eval import episode_io

    hp = ROOT / "local_eval" / "holdout_v2.json"
    samples = json.loads(hp.read_text(encoding="utf-8"))["samples"]
    picked = [s for s in samples if str(s.get("tier", "")).startswith("unseen")][:n]
    out = []
    for s in picked:
        f0 = episode_io.read_frames(s["dataset"], int(s["episode_index"]), int(s.get("start", 0)), 1)[0]
        out.append((s["sample_id"], f0))
    return out


# ---------------------------------------------------------------- A. VAE 상한
def check_vae(n: int) -> dict:
    """GT 프레임 → VAE encode→decode → 선명도 유지율·PSNR. gen(320x512) vs native 비교."""
    import torch
    from diffusers.models import AutoencoderKL

    from src.data import transforms as T

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae").to(dev).eval()
    sf = vae.config.scaling_factor

    def roundtrip(img_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img_u8).float().div(127.5).sub(1.0).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode().mul(sf)
            y = vae.decode(z.div(sf)).sample
        y = y.clamp(-1, 1).add(1).mul(127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        return y

    rows = []
    for sid, f0 in _holdout_frames(n):
        gen = T.final_to_gen_target(f0, T.RES_A)[0]        # (320,512,3) 킷 레터박스 규격
        for tag, src in (("gen_320x512", gen), ("native_%dx%d" % f0.shape[:2], f0)):
            rec = roundtrip(src)
            s0, s1 = _lap_var(_gray(src)), _lap_var(_gray(rec))
            rows.append({"sample": sid, "res": tag, "sharp_src": s0, "sharp_rec": s1,
                         "keep_pct": 100.0 * s1 / max(s0, 1e-9), "psnr": _psnr(src, rec)})
            print("  %-28s %-16s 선명도 %8.0f → %8.0f (유지 %5.1f%%)  PSNR %.1f dB"
                  % (sid[:28], tag, s0, s1, rows[-1]["keep_pct"], rows[-1]["psnr"]), flush=True)

    agg = {}
    for res in sorted({r["res"] for r in rows}):
        sub = [r for r in rows if r["res"] == res]
        agg[res] = {"keep_pct": float(np.mean([r["keep_pct"] for r in sub])),
                    "psnr": float(np.mean([r["psnr"] for r in sub]))}
    return {"rows": rows, "agg": agg}


# ------------------------------------------------- B. 해상도 · pos_embed 대조
def check_resolution(ckpt: str | None) -> dict:
    """ckpt pos_embed 토큰 수 → 원 학습 해상도 역산, 현 gen_hw(320x512)와 대조."""
    import torch

    from src.models.irasim_runtime import VENDOR, _add_vendor_path, build_vendor_cfg

    _add_vendor_path()
    from models import get_models

    cfg = build_vendor_cfg(gen_hw=(320, 512), num_frames=16)
    model = get_models(cfg)
    msd = model.state_dict()
    out = {"gen_hw": [320, 512], "latent_size": list(cfg.latent_size)}

    # 벤더 config가 명시한 학습 해상도
    try:
        import yaml
        d = yaml.safe_load((VENDOR / "configs" / "base" / "data.yaml").read_text())
        out["vendor_data_yaml_video_size"] = d.get("video_size")
    except Exception as e:  # noqa: BLE001
        out["vendor_data_yaml_video_size"] = f"(읽기 실패: {e})"

    def tokens_to_hw(tok: int, patch: int = 2, vae_div: int = 8) -> str:
        """토큰 수 → 가능한 픽셀 해상도 후보(4:5 / 4:3 / 16:10 등 흔한 비율)."""
        cands = []
        for h in range(64, 1025, 16):
            w = tok * (patch * vae_div) ** 2 // h
            if 64 <= w <= 2048 and h * w == tok * (patch * vae_div) ** 2:
                cands.append(f"{h}x{w}")
        return ", ".join(cands[:6]) or "(정수해 없음)"

    if ckpt and Path(ckpt).exists():
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = raw["ema"] if isinstance(raw, dict) and "ema" in raw else raw.get("model", raw)
        out["ckpt"] = str(ckpt)
        out["ckpt_is_ema"] = bool(isinstance(raw, dict) and "ema" in raw)
        for k in ("pos_embed", "temp_embed"):
            if k in sd and k in msd:
                ck, mk = tuple(sd[k].shape), tuple(msd[k].shape)
                out[k] = {"ckpt": list(ck), "model": list(mk), "match": ck == mk}
                if k == "pos_embed":
                    out["ckpt_native_hw_candidates"] = tokens_to_hw(ck[1])
                    out["model_hw_tokens"] = mk[1]
                print("  %-10s ckpt%s vs model%s  →  %s" % (k, ck, mk, "일치" if ck == mk else "★불일치"))
        if "pos_embed" in out and not out["pos_embed"]["match"]:
            print("  ckpt 토큰 %d → 원 학습 해상도 후보: %s" % (out["pos_embed"]["ckpt"][1],
                                                              out["ckpt_native_hw_candidates"]))
            print("  현재 %d 토큰(320x512)으로 생성 → 위치 임베딩 드롭(모델 자체값 사용)" % out["model_hw_tokens"])
    else:
        out["ckpt"] = "(경로 없음 — --ckpt 로 지정)"
    return out


# ------------------------------------------------------------ C. 노이즈 스케줄
def check_schedule() -> dict:
    """terminal SNR 확인. 0이 아니면 마지막 스텝에 신호가 남아 저대비·뿌연 출력 경향."""
    from src.models.irasim_runtime import _add_vendor_path

    _add_vendor_path()
    from diffusion import create_mask_diffusion

    d = create_mask_diffusion(timestep_respacing="", learn_sigma=True)
    acp = np.asarray(d.alphas_cumprod, dtype=np.float64)
    term_snr = float(acp[-1] / max(1.0 - acp[-1], 1e-12))
    out = {"num_timesteps": int(d.num_timesteps), "beta_first": float(d.betas[0]),
           "beta_last": float(d.betas[-1]), "alphas_cumprod_last": float(acp[-1]),
           "terminal_snr": term_snr, "zero_terminal_snr": bool(acp[-1] < 1e-4)}
    print("  timesteps %d · beta %.5f→%.5f · alpha_cumprod[-1] %.6f · terminal SNR %.4f → %s"
          % (out["num_timesteps"], out["beta_first"], out["beta_last"], out["alphas_cumprod_last"],
             term_snr, "zero-terminal-SNR 적용" if out["zero_terminal_snr"] else "★미적용(잔여 신호 있음)"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4, help="VAE 상한 측정에 쓸 홀드아웃 샘플 수")
    ap.add_argument("--ckpt", default="ckpts/bridge_frame_ada_0300000.pt", help="pos_embed 대조용 ckpt")
    ap.add_argument("--skip-vae", action="store_true", help="A(VAE) 건너뛰기 — GPU 없을 때")
    args = ap.parse_args()

    res: dict = {}
    print("\n[B] 해상도 · pos_embed 대조")
    res["resolution"] = check_resolution(args.ckpt)
    print("\n[C] 노이즈 스케줄 (terminal SNR)")
    res["schedule"] = check_schedule()
    if not args.skip_vae:
        print("\n[A] VAE 상한 (GT 프레임 encode→decode만)")
        res["vae"] = check_vae(args.n)
        print("\n  요약:", json.dumps(res["vae"]["agg"], ensure_ascii=False))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "diag_blur.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n→ results/diag_blur.json")


if __name__ == "__main__":
    main()
