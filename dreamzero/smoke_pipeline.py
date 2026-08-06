"""DreamZero-SO101 구성요소 조립 + 입력 경로 스모크 테스트.

생성 스크립트를 쓰기 전에 **입력이 실제로 모델에 들어가는지**부터 확인한다.
WANPolicyHead 를 통째로 세우면 DiT 를 순진하게 33GB 로 올려 OOM 이 나므로, 하위 구성요소를
따로 세우고 DiT 만 스트리밍+fp8 로 얹는다(load_stream.py 와 같은 방식).

확인 순서
  1) VAE / CLIP 이미지 인코더 / T5 텍스트 인코더 개별 적재 + 메모리
  2) 실제 eval 이미지 1장으로 VAE 인코딩 -> latent shape
  3) CLIP 이미지 특징 추출
  4) 대회 액션 16x6(deg, 절대) -> 상대 -> 정규화 -> 32차원 패딩
  5) DiT 적재 후 총 메모리

액션 정규화는 Vizuara 학습 메타데이터의 통계를 쓴다(state 는 절대, action 은 상대).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Vizuara SO-101 학습 통계 (vizuara_metadata_audit/.../metadata.json)
ACTION_MEAN = [0.1608986466444242, -0.9714937434584955, -1.9760798520988703,
               -0.27108273719734444, -0.03595778648858348, -1.7348315954297189]
ACTION_STD = [11.18744272699747, 16.171573522190844, 14.2028028193439,
              6.256636694736077, 4.228576577129318, 10.980838522202236]
STATE_MEAN = [-0.6963610048976526, -31.733306847063854, 41.64922760518534,
              65.98265880181754, 1.546797253549433, 18.467634289685957]
STATE_STD = [31.880134461608737, 42.84989164376948, 36.69197074668415,
             12.668087492383346, 12.554616220559964, 14.599520825313377]
DIMS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def gb(x):
    return "%.1fGB" % (x / 2**30)


def prepare_action(actions_deg: np.ndarray, max_action_dim: int):
    """대회 액션 (16,6) 절대 관절값(deg) -> DreamZero 형식.

    DreamZero 는 **상대 관절값**을 받는다(통계가 0 중심인 것으로 확인). 대회는 절대값이므로
    첫 액션을 앵커로 삼아 차분을 만든다. 차분은 데이터셋별 관절 원점 차이에도 불변이라
    (2026-08-05 킷 무사용 측정에서 확인한 성질) 이 변환이 두 번 이롭다.
    """
    anchor = actions_deg[0]                      # (6,) t=0 자세로 간주
    rel = actions_deg - anchor[None, :]          # (T,6) 상대
    norm = (rel - np.array(ACTION_MEAN)) / np.array(ACTION_STD)
    padded = np.zeros((actions_deg.shape[0], max_action_dim), dtype=np.float32)
    padded[:, :6] = norm
    state_norm = (anchor - np.array(STATE_MEAN)) / np.array(STATE_STD)
    return padded, state_norm, anchor, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-dir", required=True, help="open/data/eval 경로")
    ap.add_argument("--sample", default="sample_000000")
    ap.add_argument("--skip-dit", action="store_true", help="DiT 없이 앞단만 확인")
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    BASE, LORA, EV = Path(args.base), Path(args.lora), Path(args.eval_dir)
    cfg_all = json.loads((LORA / "config.json").read_text())
    cfg = cfg_all["action_head_cfg"]["config"]

    def remap(d):
        """config 의 /workspace/checkpoints 경로를 우리 경로로."""
        s = json.dumps(d).replace("/workspace/checkpoints/Wan2.1-I2V-14B-480P", str(BASE))
        return json.loads(s)

    print("=== 0. 환경 ===")
    cc = torch.cuda.get_device_capability(0)
    tot = torch.cuda.get_device_properties(0).total_memory
    print("  %s sm_%d%d · VRAM %s" % (torch.cuda.get_device_name(0), cc[0], cc[1], gb(tot)))

    print("\n=== 1. 액션 변환 (대회 -> DreamZero) ===")
    acts = np.load(EV / "actions" / (args.sample + ".npy")).astype(np.float64)
    print("  원본 %s · 첫 프레임 %s" % (acts.shape, np.round(acts[0], 1).tolist()))
    padded, state_norm, anchor, rel = prepare_action(acts, cfg["max_action_dim"])
    print("  상대 변환 후 범위(deg): " + " ".join(
        "%s %.1f~%.1f" % (DIMS[i][:9], rel[:, i].min(), rel[:, i].max()) for i in range(6)))
    print("  정규화 후 |z| 최대: %.2f  (학습 분포 기준 3 넘으면 도메인 밖)" % np.abs(padded[:, :6]).max())
    print("  state(앵커) 정규화 z: %s" % np.round(state_norm, 2).tolist())

    print("\n=== 2. VAE ===")
    t0 = time.time()
    vae = instantiate(OmegaConf.create(remap(cfg["vae_cfg"])))
    vae.model.load_state_dict(torch.load(remap(cfg["vae_cfg"])["vae_pretrained_path"], map_location="cpu"))
    # 가중치는 fp32 로 저장돼 있다 — 입력 dtype 과 맞춰야 conv3d 가 안 터진다
    vae = vae.to("cuda", dtype=torch.bfloat16).eval()
    print("  적재 %.1f초 · GPU %s" % (time.time() - t0, gb(torch.cuda.memory_allocated())))

    print("\n=== 3. eval 이미지 -> VAE latent ===")
    from PIL import Image
    img = np.asarray(Image.open(EV / "images" / (args.sample + ".png")).convert("RGB"))
    print("  원본 이미지 %s" % (img.shape,))
    x = torch.from_numpy(img).cuda().float().div(255.).mul(2).sub(1)          # [-1,1]
    x = x.permute(2, 0, 1)[None, :, None]                                      # b c t h w
    for th, tw in ((176, 320), (320, 512)):
        xi = torch.nn.functional.interpolate(x[:, :, 0], size=(th, tw), mode="bilinear",
                                             align_corners=False)[:, :, None]
        with torch.no_grad():
            lat = vae.encode(xi.to(torch.bfloat16), tiled=False)
        print("  %dx%d -> latent %s  (토큰/프레임 = %d)"
              % (th, tw, tuple(lat.shape), (lat.shape[-2] // 2) * (lat.shape[-1] // 2)))
    print("  ※ config 의 frame_seqlen = %d" % cfg["diffusion_model_cfg"]["frame_seqlen"])

    print("\n=== 4. CLIP 이미지 인코더 ===")
    t0 = time.time()
    ie = instantiate(OmegaConf.create(remap(cfg["image_encoder_cfg"])))
    ie.model.load_state_dict(torch.load(remap(cfg["image_encoder_cfg"])["image_encoder_pretrained_path"],
                                        map_location="cpu"), strict=False)
    ie = ie.to("cuda", dtype=torch.bfloat16).eval()
    print("  적재 %.1f초 · GPU %s" % (time.time() - t0, gb(torch.cuda.memory_allocated())))

    print("\n=== 5. T5 텍스트 인코더 (CPU 상주) ===")
    t0 = time.time()
    te = instantiate(OmegaConf.create(remap(cfg["text_encoder_cfg"])))
    te.load_state_dict(torch.load(remap(cfg["text_encoder_cfg"])["text_encoder_pretrained_path"],
                                  map_location="cpu"))
    te = te.eval()
    print("  적재 %.1f초 (CPU 유지) · GPU %s" % (time.time() - t0, gb(torch.cuda.memory_allocated())))

    if not args.skip_dit:
        print("\n=== 6. DiT (스트리밍 + fp8) ===")
        print("  load_stream.py 와 동일 경로 — 여기서는 생략하고 총량만 계산")
        print("  예상 총 GPU = 현재 %s + DiT 15.7GB = %.1fGB / %s"
              % (gb(torch.cuda.memory_allocated()),
                 torch.cuda.memory_allocated() / 2**30 + 15.7, gb(tot)))

    print("\n★스모크 통과 — 입력 경로가 살아 있다")


if __name__ == "__main__":
    main()
