"""판독기 관문 — "영상만 보고 팔 자세를 읽을 수 있는가"를 손실에 연결하기 전에 먼저 재본다.

**왜 이 시험이 필요한가.** 태양님 제안(state-estimator aux loss)은 영상→관절각 판독기를 만들어
확산 학습의 보조 손실로 쓰자는 것이다. 그런데 판독기가 **배경을 보고 데이터셋을 알아맞힌 뒤
그 데이터셋의 평균 자세를 뱉기만 해도** 오차는 작아 보인다. 그런 판독기를 손실에 연결하면
모델은 "팔을 그 데이터셋 평균 자리에 그려라"만 배운다 — 프레임별 정보가 없기 때문이다.

**그래서 절대값이 아니라 기준선 대비로 판정한다.**

  ① 전체 평균        아무것도 안 보고 찍기 (z 정규화 공간에서 0)
  ② 데이터셋별 평균  배경만 보고 찍기          ← **이걸 못 이기면 판독기는 장면 분류기다**
  ③ 첫 프레임 유지   팔이 안 움직인다고 찍기    ← 이걸 이겨야 '움직임을 따라간다'는 뜻

②를 확실히 이겨야 "팔을 실제로 본다", ③까지 이겨야 "시간에 따른 움직임을 읽는다"이다.

**절대각과 변화량을 같이 잰다.** [pan_origin.py](pan_origin.py) 실측으로 관절각→화면 대응이
데이터셋마다 다르다는 것을 확인했다(원점 표준편차 42.4도, 기울기 부호 22곳 양수/55곳 음수).
절대각이 안 되면 변화량(Δ) 판이 대안이다 — Δ 는 원점이 어디든 영향받지 않는다.

**에피소드 단위로 나눈다.** 한 에피소드 안에서 클립을 쪼개 나누면 이웃 프레임이 양쪽에 들어가
사실상 같은 그림을 외운 것이 검증에 새어 든다.

eval 은 전혀 보지 않는다 — 학습 데이터 안에서만 닫힌 측정이다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 데이터 수집

def load_stats(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(p.read_text())
    return np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32)


def episode_clips(vid: Path, pq: Path, n_clip: int, T: int, hw: tuple[int, int], rng: random.Random):
    """에피소드 하나에서 클립 n_clip 개. 반환 [(frames[T,h,w,3] uint8, state[T,6] float32), ...]"""
    import imageio.v3 as iio
    import pandas as pd
    import cv2

    # ★타깃은 `action` 이다. 킷 데이터셋이 parquet 에서 action 만 읽어 배치에 observation.state 가
    #   아예 없고(lerobot_so100.py:387), 채점기도 video→action 관례다. 보조손실이 실제로 쓸
    #   그 양을 그대로 재야 관문이 의미가 있다. so100_action_statistics.json 도 action 통계다.
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return []
    col = "action" if "action" in df.columns else "observation.state"
    st = np.stack([np.asarray(s, np.float32)[:6] for s in df[col].values])
    n = len(st)
    if n < T:
        return []

    starts = sorted(rng.sample(range(0, n - T + 1), min(n_clip, n - T + 1)))
    need = {i for s in starts for i in range(s, s + T)}
    hi = max(need)

    keep: dict[int, np.ndarray] = {}
    h, w = hw
    try:
        for i, fr in enumerate(iio.imiter(vid, plugin="pyav")):
            if i in need:
                keep[i] = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            if i >= hi:
                break
    except Exception:
        return []
    if len(keep) < len(need):
        return []

    out = []
    for s in starts:
        out.append((np.stack([keep[i] for i in range(s, s + T)]), st[s : s + T]))
    return out


def collect(root: Path, n_ep: int, n_clip: int, T: int, hw: tuple[int, int], seed: int):
    """데이터셋마다 에피소드를 뽑아 학습/검증으로 **에피소드 단위** 분할."""
    rng = random.Random(seed)
    dss = sorted({p.parent for p in root.glob("*/*/data")})
    print(f">>> 데이터셋 {len(dss)}개", flush=True)
    tr, va = [], []
    t0 = time.time()
    for k, ds in enumerate(dss, 1):
        vids = sorted(ds.glob("videos/*/*/*.mp4"))
        if not vids:
            continue
        cam = vids[0].parent                       # 카메라가 여럿이면 첫 번째만
        eps = sorted(cam.glob("*.mp4"))
        eps = eps[:: max(1, len(eps) // n_ep)][:n_ep]
        if len(eps) < 2:
            continue
        n_val = max(1, len(eps) // 5)              # 에피소드의 20% 를 검증으로
        for j, v in enumerate(eps):
            cand = list(ds.glob(f"data/*/{v.stem}.parquet"))
            if not cand:
                continue
            for fr, st in episode_clips(v, cand[0], n_clip, T, hw, rng):
                (va if j < n_val else tr).append((k - 1, fr, st))
        if k % 16 == 0:
            print(f"  [{k}/{len(dss)}] 학습 {len(tr)} · 검증 {len(va)} · {time.time()-t0:.0f}초", flush=True)
    return tr, va, len(dss)


# ---------------------------------------------------------------- 판독기

class Residual3DBlock(nn.Module):
    def __init__(self, ci, co, stride=(1, 2, 2)):
        super().__init__()
        g = lambda c: min(8, c)
        self.c1 = nn.Conv3d(ci, co, 3, stride, 1)
        self.n1 = nn.GroupNorm(g(co), co)
        self.c2 = nn.Conv3d(co, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(g(co), co)
        self.skip = nn.Conv3d(ci, co, 1, stride) if (ci != co or stride != (1, 1, 1)) else nn.Identity()

    def forward(self, x):
        h = nn.functional.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return nn.functional.silu(h + self.skip(x))


class KitShapedProbe(nn.Module):
    """킷의 `SO100ActionExtractor` 와 **같은 모양** — Conv3d 앞단 + 양방향 GRU.

    2D 판(StateProbe)이 Δ 에서 완전히 실패한 원인이 여기 있다고 본다: 2D 인코더는 프레임 하나를
    벡터 하나로 짜부라뜨린 **뒤에야** 시간축을 보므로, 그 압축에서 프레임 간 변위가 날아간다.
    킷은 첫 층부터 3D 합성곱이라 움직임이 특징에 직접 실린다. 가중치는 안 쓰고 구조만 맞춘다.
    """

    def __init__(self, out_dim: int = 6, base: int = 32, mults=(1, 2, 4), hidden: int = 128):
        super().__init__()
        c0 = base * mults[0]
        self.stem = nn.Sequential(
            nn.Conv3d(3, c0, (3, 7, 7), (1, 2, 2), (1, 3, 3)),
            nn.GroupNorm(min(8, c0), c0), nn.SiLU())
        chs = [base * m for m in mults]
        self.enc = nn.Sequential(*[Residual3DBlock(a, b) for a, b in zip(chs, chs[1:])])
        self.gru = nn.GRU(chs[-1], hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 256),
                                  nn.SiLU(), nn.Linear(256, out_dim))

    def forward(self, x):                      # x [B,T,3,H,W]
        v = x.permute(0, 2, 1, 3, 4)           # [B,3,T,H,W]
        f = self.enc(self.stem(v)).mean((-1, -2))   # [B,C,T]
        h, _ = self.gru(f.transpose(1, 2))     # [B,T,2*hidden]
        return self.head(h)


class StateProbe(nn.Module):
    """프레임별 CNN → 시간축 conv → [T,6]. 맨바닥부터 학습한다(사전학습 가중치 없음 = 라이선스 문제 없음)."""

    def __init__(self, out_dim: int = 6, width: int = 48):
        super().__init__()
        c = width
        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.SiLU())
        self.enc = nn.Sequential(blk(3, c), blk(c, c * 2), blk(c * 2, c * 4),
                                 blk(c * 4, c * 4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.tmp = nn.Sequential(nn.Conv1d(c * 4, c * 4, 5, padding=2), nn.GroupNorm(8, c * 4), nn.SiLU(),
                                 nn.Conv1d(c * 4, c * 4, 5, padding=2), nn.GroupNorm(8, c * 4), nn.SiLU())
        self.head = nn.Conv1d(c * 4, out_dim, 1)

    def forward(self, x):                      # x [B,T,3,H,W] (-1~1)
        b, t = x.shape[:2]
        f = self.enc(x.flatten(0, 1)).view(b, t, -1).transpose(1, 2)   # [B,C,T]
        return self.head(self.tmp(f)).transpose(1, 2)                  # [B,T,6]


# ---------------------------------------------------------------- 학습·평가

def to_tensors(items, mean, std, dstd, device):
    ds_id = torch.tensor([i for i, _, _ in items], dtype=torch.long)
    fr = torch.from_numpy(np.stack([f for _, f, _ in items]))           # [N,T,H,W,3] uint8
    st = torch.from_numpy(np.stack([s for _, _, s in items]))           # [N,T,6]
    z = (st - torch.from_numpy(mean)) / torch.from_numpy(std)           # 절대값 z 정규화
    d = torch.diff(st, dim=1, prepend=st[:, :1]) / torch.from_numpy(dstd)   # 변화량 정규화
    return ds_id, fr, z.float(), d.float()


def mae(a, b):
    return (a - b).abs().mean().item()


def run(target: str, tr, va, device, epochs, bs, lr, log, arch="conv2d"):
    """target: 'abs'(절대각 z) 또는 'delta'(변화량)"""
    ds_tr, fr_tr, z_tr, d_tr = tr
    ds_va, fr_va, z_va, d_va = va
    y_tr, y_va = (z_tr, z_va) if target == "abs" else (d_tr, d_va)

    # ── 기준선 세 가지 (검증 클립에서)
    zero = torch.zeros_like(y_va)
    b_global = mae(zero, y_va)                                   # ① 전체 평균 (z 공간에서 0)
    per = torch.zeros(int(ds_tr.max()) + 1, y_tr.shape[-1])      # ② 데이터셋별 평균 (학습분에서)
    for i in range(per.shape[0]):
        m = ds_tr == i
        if m.any():
            per[i] = y_tr[m].mean((0, 1))
    b_perds = mae(per[ds_va][:, None, :].expand_as(y_va), y_va)
    b_hold = mae(y_va[:, :1].expand_as(y_va), y_va)              # ③ 첫 프레임 유지 (GT 사용)

    log(f"[{target}] 기준선 — 전체평균 {b_global:.4f} · 데이터셋평균 {b_perds:.4f} · 첫프레임유지 {b_hold:.4f}")

    cls = KitShapedProbe if arch == "kit" else StateProbe
    model = cls(out_dim=y_tr.shape[-1]).to(device)
    log(f"[{target}] 구조 {arch} · 파라미터 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(fr_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * max(1, n // bs))
    best = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = cnt = 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i : i + bs]
            x = fr_tr[idx].to(device).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
            y = y_tr[idx].to(device)
            loss = nn.functional.l1_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); cnt += 1
        model.eval()
        errs, per_joint = [], []
        with torch.no_grad():
            for i in range(0, len(fr_va), bs):
                x = fr_va[i : i + bs].to(device).permute(0, 1, 4, 2, 3).float().div_(127.5).sub_(1.0)
                p = model(x).cpu()
                errs.append((p - y_va[i : i + bs]).abs())
        e = torch.cat(errs)
        v = e.mean().item()
        best = min(best, v)
        log(f"[{target}] epoch {ep+1}/{epochs} · 학습 {tot/max(cnt,1):.4f} · 검증 {v:.4f}"
            + ("  ← 최저" if v == best else ""))
        if ep == epochs - 1:
            per_joint = e.mean((0, 1)).tolist()

    log(f"[{target}] ★결과 판독기 {best:.4f}  vs  데이터셋평균 {b_perds:.4f}  vs  첫프레임유지 {b_hold:.4f}")
    log(f"[{target}] 관절별 오차 {[round(x,4) for x in per_joint]}")
    verdict = ("통과 — 데이터셋 평균을 이겼고 움직임도 따라간다" if best < b_perds * 0.9 and best < b_hold
               else "부분 — 데이터셋 평균은 이겼으나 움직임 추적은 못한다" if best < b_perds * 0.9
               else "탈락 — 배경만 보고 찍는 것과 다르지 않다")
    log(f"[{target}] ★판정: {verdict}")
    return {"target": target, "probe": best, "global": b_global, "per_dataset": b_perds,
            "hold": b_hold, "per_joint": per_joint, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--delta-stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=10, help="데이터셋당 에피소드")
    ap.add_argument("--clips", type=int, default=3, help="에피소드당 클립")
    ap.add_argument("--traj-len", type=int, default=16)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=("conv2d", "kit"), default="kit",
                    help="kit = Conv3d 앞단 + 양방향 GRU (킷 추출기와 같은 모양)")
    ap.add_argument("--cache", default=None,
                    help="수집한 클립을 여기 저장/재사용 — 구조만 바꿔 재시도할 때 수집 10분을 건너뛴다")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    logf = open(args.out + ".log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    mean, std = load_stats(Path(args.stats))
    _, dstd = load_stats(Path(args.delta_stats))
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        tr_raw = list(zip(z["tr_id"], z["tr_fr"], z["tr_st"]))
        va_raw = list(zip(z["va_id"], z["va_fr"], z["va_st"]))
        n_ds = int(z["n_ds"])
        log(f">>> 캐시 재사용: {cache} (수집 건너뜀)")
    else:
        log(f">>> 수집 시작 · {args.episodes}에피소드/데이터셋 × {args.clips}클립 · {args.height}x{args.width}")
        tr_raw, va_raw, n_ds = collect(Path(args.root), args.episodes, args.clips,
                                       args.traj_len, (args.height, args.width), args.seed)
        if cache:
            np.savez(cache,
                     tr_id=np.array([a for a, _, _ in tr_raw]), tr_fr=np.stack([b for _, b, _ in tr_raw]),
                     tr_st=np.stack([c for _, _, c in tr_raw]),
                     va_id=np.array([a for a, _, _ in va_raw]), va_fr=np.stack([b for _, b, _ in va_raw]),
                     va_st=np.stack([c for _, _, c in va_raw]), n_ds=n_ds)
            log(f">>> 캐시 저장: {cache}")
    log(f">>> 클립 학습 {len(tr_raw)} · 검증 {len(va_raw)} (에피소드 단위 분할)")
    if len(va_raw) < 50 or len(tr_raw) < 200:
        log("★표본이 너무 적다 — 중단"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = to_tensors(tr_raw, mean, std, dstd, device)
    va = to_tensors(va_raw, mean, std, dstd, device)
    del tr_raw, va_raw

    res = [run(t, tr, va, device, args.epochs, args.batch_size, args.lr, log, args.arch)
           for t in ("abs", "delta")]
    json.dump({"n_dataset": n_ds, "arch": args.arch, "results": res},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    log(f">>> 저장: {args.out}")


if __name__ == "__main__":
    main()
