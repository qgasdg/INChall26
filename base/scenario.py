"""조건 이미지는 그대로 두고 **액션만 사람이 설계한 것으로** 바꿔 생성한다.

A/B 테스트(`abtest.py`)는 실제 궤적을 뒤집어 썼다. 여기서는 한 발 더 나아가
**한 관절씩만 움직이는 인공 궤적**을 넣는다. 무엇이 화면에 나타나야 하는지가 미리 정해지므로
"액션이 반영됐다"를 훨씬 좁게 판정할 수 있다.

**규칙 두 가지.**
1. 모든 시나리오는 **조건 이미지의 자세(GT 프레임 0)에서 출발**한다 — 학습에서 본 적 없는
   "조건과 액션이 어긋난 입력"을 만들지 않기 위해서다.
2. 값은 그 데이터셋에서 **실제로 관측된 관절 범위 안**에 둔다.

Δ 는 절대값에서 다시 계산한다(`data12.Act12Dataset` 과 같은 식).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")

BASELINE = Path(os.environ.get("FT_BASELINE", os.path.expanduser("~/ft/kit/baseline")))
KIT = BASELINE / "challenge_kit"
for p in (KIT / "libs/dynamicrafter", KIT / "src", BASELINE / "shared_libs/video_utils", KIT,
          Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from lvdm.models.samplers.ddim import DDIMSampler  # noqa: E402
from lvdm.utils.train import get_model  # noqa: E402
from lvdm.utils.utils import instantiate_from_config  # noqa: E402

from generate import save_grid  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
T = 16


def ramp(a: float, b: float, n: int = T) -> np.ndarray:
    return np.linspace(a, b, n)


def build_scenarios(p0: np.ndarray) -> dict[str, np.ndarray]:
    """p0: [6] 조건 이미지의 관절각(도) → 이름 → [16,6] 절대 궤적(도).

    duck 관측 범위: pan −42.7~46.5 · lift 34.0~189.6 · elbow 9.0~178.2 ·
                    wrist_flex −17.2~98.6 · wrist_roll −109.9~−25.7 · gripper 0~50.7
    """
    def hold():
        return np.tile(p0, (T, 1))

    s = {}

    # ① 정지 — 16프레임 내내 같은 자세. Δ 가 전부 0 이다.
    s["S1_정지"] = hold()

    # ② 팬만 왼쪽 끝까지 — 다른 관절은 고정. 화면에서 팔이 한쪽으로 쓸려야 한다.
    a = hold(); a[:, 0] = ramp(p0[0], -42.0)
    s["S2_팬_왼쪽"] = a

    # ③ 팬만 오른쪽 끝까지 — ②의 반대 방향.
    a = hold(); a[:, 0] = ramp(p0[0], 46.0)
    s["S3_팬_오른쪽"] = a

    # ④ 그리퍼만 열고 닫기 — 팔은 제자리. 집게 끝만 변해야 한다.
    a = hold()
    a[:, 5] = np.concatenate([ramp(p0[5], 50.0, 8), ramp(50.0, p0[5], 8)])
    s["S4_그리퍼만"] = a

    # ⑤ 팔만 아래로 뻗기 — lift·elbow 를 낮춘다. 팬은 고정이라 좌우 이동이 없어야 한다.
    a = hold()
    a[:, 1] = ramp(p0[1], 55.0)
    a[:, 2] = ramp(p0[2], 45.0)
    s["S5_아래로"] = a

    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/base/base-03.yaml"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--idx", type=int, default=0, help="조건 이미지를 가져올 val 클립")
    ap.add_argument("--out", default=os.path.expanduser("~/ft/out/scenario"))
    ap.add_argument("--cfg", type=float, default=None, help="액션 CFG 배율")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cfg = OmegaConf.load(args.config)
    cfg.model.params.use_ema = False
    cfg.model.params.resume_unet = None
    print(">>> 모델 구성 중", flush=True)
    model = get_model(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    _, unexpected = model.load_state_dict(sd, strict=False)
    print(f">>> 학습본 적재: {args.ckpt} (안 실린 키 {len(unexpected)}개)", flush=True)
    model.to(device).eval()
    sampler = DDIMSampler(model)

    ddim_kwargs = OmegaConf.to_container(cfg.ddim_kwargs, resolve=True)
    if args.cfg is not None:
        ddim_kwargs["unconditional_guidance_scale"] = args.cfg

    data = instantiate_from_config(cfg.data); data.setup()
    item = data.val_dataset[args.idx]
    gt = item["video"].unsqueeze(0).to(device)

    st = json.loads(Path(cfg.data.params.action_stats_path).read_text())
    ds = json.loads(Path(cfg.data.params.delta_stats_path).read_text())
    mean = np.array(st["mean"], dtype=np.float32); std = np.array(st["std"], dtype=np.float32)
    scale = torch.tensor(std / np.array(ds["std"], dtype=np.float32)).float().to(device)

    real_deg = item["act"].numpy()[:, :6] * std + mean
    p0 = real_deg[0].copy()
    print(f"\n조건 이미지 자세(도): " + " · ".join(f"{j}={v:.1f}" for j, v in zip(JOINTS, p0)))

    cases = {"S0_실제": real_deg, **build_scenarios(p0)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_grid(gt[0], out / "GT.png")

    def to12(deg: np.ndarray) -> torch.Tensor:
        z = torch.tensor((deg - mean) / std, dtype=torch.float32, device=device)
        dz = torch.diff(z, dim=0, prepend=z[:1])
        return torch.cat([z, dz * scale], dim=-1)

    def to255(x):
        return (x.clamp(-1, 1) + 1) * 127.5

    print(f"\n{'시나리오':<14}{'프레임간':>10}{'누적변위':>10}{'S1대비':>9}{'초':>7}")
    print("-" * 52)
    outs = {}
    for name, deg in cases.items():
        t0 = time.time()
        torch.manual_seed(args.seed)
        v = torch.zeros_like(gt); v[:, :, 0] = gt[:, :, 0]        # 조건은 항상 GT 프레임 0
        batch = {"video": v, "act": to12(deg).unsqueeze(0), "caption": [""],
                 "fps": torch.full((1,), args.fps, dtype=torch.long, device=device),
                 "frame_stride": torch.full((1,), args.fps, dtype=torch.long, device=device),
                 "start_idx": torch.zeros(1, dtype=torch.long, device=device)}
        z, c, uc, cond_mask, _, kw = model.prepare_batch_for_inference(batch)
        sk = dict(ddim_kwargs); sk.update(kw)
        steps = sk.pop("ddim_steps")
        shape = (model.channels, model.temporal_length, *model.image_size)
        with torch.no_grad(), torch.cuda.amp.autocast():
            s, _ = sampler.sample(steps, batch_size=1, shape=shape, conditioning=c,
                                  unconditional_conditioning=uc, mask=cond_mask, x0=z, **sk)
        gen = model.decode_first_stage(s)
        save_grid(gen[0], out / f"{name}.png")
        outs[name] = gen
        g = to255(gen[0]).permute(1, 0, 2, 3)
        rel = (to255(gen) - to255(outs["S1_정지"])).abs().mean().item() if "S1_정지" in outs else float("nan")
        print(f"{name:<14}{(g[1:]-g[:-1]).abs().mean():10.3f}{(g[-1]-g[0]).abs().mean():10.3f}"
              f"{rel:9.3f}{time.time()-t0:7.1f}")

    print(f"\n=== 시나리오끼리 차이 (MAE, 0~255) ===")
    ks = list(outs)
    print(f"{'':<14}" + "".join(f"{k:>14}" for k in ks))
    for a in ks:
        row = "".join(f"{(to255(outs[a])-to255(outs[b])).abs().mean().item():14.3f}" for b in ks)
        print(f"{a:<14}{row}")
    print(f"\n그림: {out}")


if __name__ == "__main__":
    main()
