"""액션 조건 영상 생성 — 주어진 액션으로 비디오만 디노이징한다.

DreamZero 는 원래 비디오와 액션을 **함께 예측**하는 정책 모델이다. 우리 과제는 반대로
액션이 입력으로 주어지고 비디오만 만들면 된다. 다행히 모델이 두 스트림에 각자의
타임스텝(`timestep`, `timestep_action`)을 주도록 설계돼 있어 개조가 아니라 호출 방식 문제다.

lazy_joint_video_action(370줄)을 복사하지 않고 세 지점만 패치한다.
  1) generate_noise 가 액션 모양을 요청하면 **노이즈 대신 정답 액션**을 돌려준다
  2) 액션 스케줄러의 step 을 **항등**으로 (디노이징하지 않음 = 계속 깨끗)
  3) 액션 스케줄러의 timesteps 를 **전부 0** 으로 (완전히 디노이즈된 상태로 표시)

DiT 는 33GB 라 통짜로 못 올린다. CausalWanModel 만 meta 로 만들고 샤드를 흘려 넣은 뒤
fp8 로 GPU 에 올린다(load_stream.py 와 동일).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ACTION_MEAN = np.array([0.1608986466444242, -0.9714937434584955, -1.9760798520988703,
                        -0.27108273719734444, -0.03595778648858348, -1.7348315954297189])
ACTION_STD = np.array([11.18744272699747, 16.171573522190844, 14.2028028193439,
                       6.256636694736077, 4.228576577129318, 10.980838522202236])
STATE_MEAN = np.array([-0.6963610048976526, -31.733306847063854, 41.64922760518534,
                       65.98265880181754, 1.546797253549433, 18.467634289685957])
STATE_STD = np.array([31.880134461608737, 42.84989164376948, 36.69197074668415,
                      12.668087492383346, 12.554616220559964, 14.599520825313377])
H, W = 352, 640          # frame_seqlen 880 을 역산한 네이티브 해상도


def resample(a: np.ndarray, n: int) -> np.ndarray:
    """(T,D) 시계열을 n 스텝으로 선형 보간."""
    t_old = np.linspace(0.0, 1.0, a.shape[0])
    t_new = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(t_new, t_old, a[:, i]) for i in range(a.shape[1])], axis=1)


def build_head(root, base, lora, device="cuda"):
    """WANPolicyHead 를 세우되 DiT 는 meta 로 만들고 나중에 스트리밍으로 채운다."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    import groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf as WM
    from groot.vla.model.dreamzero.modules.utils import init_weights_on_device
    from groot.vla.model.dreamzero.modules.vram_management import (
        AutoWrappedLinear, AutoWrappedModule, enable_vram_management)

    base, lora = Path(base), Path(lora)
    raw = json.loads((lora / "config.json").read_text())["action_head_cfg"]["config"]
    raw = json.loads(json.dumps(raw).replace("/workspace/checkpoints/Wan2.1-I2V-14B-480P", str(base)))
    raw["skip_component_loading"] = True          # DiT 가중치 적재만 건너뛴다
    # ★peft 로 LoRA 를 주입하면 base 파라미터 이름이 q.weight -> q.base_layer.weight 로 바뀌어
    #   샤드가 안 실린다. 주입을 끄고 우리가 직접 병합한다(load_stream.py 에서 검증된 경로).
    raw["train_architecture"] = "full"

    # DiT 만 meta 로 — 그대로 두면 fp32 66GB 를 CPU 에 잡아 죽는다
    orig_instantiate = WM.instantiate

    def patched(cfg, *a, **k):
        tgt = str(cfg.get("_target_", "")) if hasattr(cfg, "get") else ""
        if "CausalWanModel" in tgt:
            with init_weights_on_device(device=torch.device("meta")):
                return orig_instantiate(cfg, *a, **k)
        return orig_instantiate(cfg, *a, **k)

    WM.instantiate = patched
    cfg_obj = OmegaConf.create(raw)
    print("  헤드 구성 중 (T5/CLIP/VAE 적재, DiT 는 meta)…", flush=True)
    head = WM.WANPolicyHead(config=WM.WANPolicyHeadConfig(**OmegaConf.to_container(cfg_obj, resolve=True)))
    WM.instantiate = orig_instantiate

    print("  DiT 스트리밍 적재 (샤드 -> LoRA 병합 -> fp8 -> GPU)…", flush=True)
    # ★CPU 에 33GB 를 모으면 헤드가 든 T5(11GB)+CLIP(4.5GB) 와 합쳐 62GB RAM 을 넘긴다.
    #   샤드 하나를 읽는 즉시 LoRA 를 병합하고 fp8 로 바꿔 GPU 로 보낸다 -> CPU 피크 = 샤드 1개.
    lw = load_file(str(lora / "model.safetensors"))
    P = "action_head.model.base_model.model."
    lw = {k[len(P):]: v for k, v in lw.items() if k.startswith(P)}
    pairs = {}
    for k, v in lw.items():
        m = re.match(r"^(.*?)\.lora_(A|B)(?:\.[^.]+)?\.weight$", k)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = v
    scale = raw["lora_alpha"] / raw["lora_rank"]
    # ★델타를 미리 펼치면 안 된다. rank-4 압축본(A,B)은 작지만 B@A 는 전체 크기라
    #   400개를 fp32 로 다 만들면 60GB 를 넘겨 OOM 이 난다. 샤드 안에서 필요할 때만 만든다.
    ab_by_key = {name + ".weight": ab for name, ab in pairs.items() if "A" in ab and "B" in ab}

    idx = json.loads((base / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    n = 0
    for s_name in sorted(set(idx.values())):
        sd = {}
        for k, v in load_file(str(base / s_name)).items():
            w = v.to(torch.float32)
            # ★LoRA 를 여기서 합치면 안 된다. 델타가 베이스의 2~6% 라 fp8(가수 3비트)
            #   반올림에 73.8% 가 먹힌다(bf16 이면 90.2% 반영). 아래에서 별도 모듈로 얹는다.
            # ★fp8 은 Linear 가중치(2차원)에만. 노름(1차원)·패치임베딩(5차원 Conv3d)은
            #   fp8 커널이 없고 코드가 .weight 를 직접 읽으므로 bf16 으로 둔다(크기도 작다).
            sd[k] = (w.to(torch.float8_e4m3fn) if w.ndim == 2 else w.to(torch.bfloat16)).cuda()
        r = head.model.load_state_dict(sd, strict=False, assign=True)
        n += len(sd) - len(r.unexpected_keys)
        del sd
        torch.cuda.empty_cache()
    # 액션·상태 모듈은 작고 정밀도가 중요하니 bf16 으로 둔다
    head.model.load_state_dict({k: v.to(torch.bfloat16).cuda() for k, v in lw.items() if ".lora_" not in k},
                               strict=False, assign=True)
    print("  베이스 %d · meta 잔여 %d · GPU %.1fGB"
          % (n, sum(v.is_meta for v in head.model.state_dict().values()),
             torch.cuda.memory_allocated() / 2**30), flush=True)

    enable_vram_management(
        head.model,
        # Linear 만 래핑한다. AutoWrappedModule 은 .weight 를 노출하지 않아 노름·Conv3d 를
        # 감싸면 모델 코드가 터진다(AutoWrappedLinear 는 nn.Linear 상속이라 안전).
        module_map={torch.nn.Linear: AutoWrappedLinear},
        module_config=dict(offload_dtype=torch.float8_e4m3fn, offload_device="cuda",
                           onload_dtype=torch.float8_e4m3fn, onload_device="cuda",
                           computation_dtype=torch.bfloat16, computation_device="cuda"))
    # 가중치는 이미 fp8 로 GPU 에 있다 — onload 는 no-op 이지만 상태 일관성을 위해 부른다
    for m in head.model.modules():
        if hasattr(m, "onload"):
            m.onload()

    # ★LoRA 를 베이스에 합치지 않고 bf16 으로 따로 얹는다. AutoWrappedLinear.forward 가
    #   out + x @ A.T @ B.T 를 계산 dtype(bf16)에서 더해준다. scale=alpha/rank=1.0 이라
    #   별도 보정이 필요 없다. 209MB 뿐이라 메모리 부담도 없다.
    mods = dict(head.model.named_modules())
    attached = 0
    for name, d in pairs.items():
        m = mods.get(name)
        if m is not None and hasattr(m, "lora_A_weights") and "A" in d and "B" in d:
            m.lora_A_weights = [d["A"].to(torch.bfloat16).cuda()]
            m.lora_B_weights = [d["B"].to(torch.bfloat16).cuda()]
            attached += 1
    print("  ★LoRA bf16 부착 %d/%d 모듈 (병합 대신)" % (attached, len(pairs)), flush=True)
    del lw, ab_by_key, pairs
    head.vae = head.vae.to(device, dtype=torch.bfloat16).eval()
    head.image_encoder = head.image_encoder.to(device, dtype=torch.bfloat16).eval()
    head.text_encoder = head.text_encoder.eval()   # ★T5(11GB)는 CPU 유지 — 아래서 한 번 쓰고 버린다
    # ★post_initialize() 는 부르지 않는다 — 모델을 bf16 으로 되돌려 33GB 가 되면서 OOM 난다.
    #   거기서 세팅하는 것 중 추론에 필요한 것만 직접 넣는다.
    head.trt_engine = None
    torch.cuda.synchronize()
    print("  ★GPU %.1fGB / %.1fGB"
          % (torch.cuda.memory_allocated() / 2**30,
             torch.cuda.get_device_properties(0).total_memory / 2**30))
    return head


def patch_action_forcing(head, clean_action):
    """액션 스트림을 디노이징하지 않고 주어진 값으로 고정한다."""
    import torch
    import groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf as WM

    orig_noise = head.generate_noise

    def noise(shape, *a, **k):
        if len(shape) == 3 and shape[1] == head.action_horizon:
            return clean_action.clone()                       # ← 노이즈 대신 정답 액션
        return orig_noise(shape, *a, **k)

    head.generate_noise = noise

    OrigSched = WM.FlowUniPCMultistepScheduler
    state = {"n": 0}

    class Identity(OrigSched):
        """액션용 — timesteps 를 0 으로 두고 step 은 아무것도 하지 않는다."""

        def set_timesteps(self, *a, **k):
            super().set_timesteps(*a, **k)
            self.timesteps = torch.zeros_like(self.timesteps)

        def step(self, model_output=None, timestep=None, sample=None, **k):
            return (sample,)

    def factory(*a, **k):
        state["n"] += 1
        cls = Identity if state["n"] % 2 == 0 else OrigSched   # 두 번째가 액션 스케줄러
        return cls(*a, **k)

    WM.FlowUniPCMultistepScheduler = factory
    return lambda: setattr(WM, "FlowUniPCMultistepScheduler", OrigSched)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--sample", default="sample_000000")
    ap.add_argument("--action-mode", default="true", choices=("true","zero","half"),
                    help="진단용. zero=움직임 없음, half=크기 절반(fps 가설 검증)")
    ap.add_argument("--state-mode", default="true", choices=("true", "zero", "mean"),
                    help="진단용. zero=상태 0, mean=학습 평균. eval 의 wrist_roll 이 z=7.85 로 "
                         "학습 분포 밖이라 상태 인코더를 오염시키는지 본다")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--prompt", default="a robot arm manipulating an object on a table")
    ap.add_argument("--blocks", type=int, default=2, help="블록 수. 1블록=latent 3, 이후 +2. latent L -> 영상 (L-1)*4+1 프레임. 2블록이면 17프레임으로 대회 16프레임을 덮는다")
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    import torch
    # 스케줄러의 multistep_uni_p_bh_update 가 step_index 마다 재컴파일된다. 추론 스텝이
    # 16 이라 기본 한도 8 을 넘겨 죽는다 — 한도를 올린다.
    torch._dynamo.config.recompile_limit = 128
    torch._dynamo.config.cache_size_limit = 128
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.feature_extraction_utils import BatchFeature

    EV = Path(args.eval_dir)
    print("=== 1. 모델 ===")
    t0 = time.time()
    head = build_head(args.root, args.base, args.lora)
    print("  %.1f초" % (time.time() - t0))

    print("\n=== 2. 입력 ===")
    img = np.asarray(Image.open(EV / "images" / (args.sample + ".png")).convert("RGB"))
    im = np.asarray(Image.fromarray(img).resize((W, H), Image.BICUBIC))
    images = torch.from_numpy(im.copy())[None, None].cuda()            # b t h w c (uint8)

    acts = np.load(EV / "actions" / (args.sample + ".npy")).astype(np.float64)
    anchor = acts[0]
    rel = acts - anchor[None, :]
    rel24 = resample(rel, head.action_horizon)                          # 16 -> 24 스텝
    norm = (rel24 - ACTION_MEAN) / ACTION_STD
    a_pad = np.zeros((head.action_horizon, head.model.action_dim), dtype=np.float32)
    a_pad[:, :6] = norm
    if args.action_mode == "zero":
        a_pad[:] = 0.0                       # 상대 이동 0 = 정지 요구
    elif args.action_mode == "half":
        a_pad[:, :6] *= 0.5                  # 학습 분포 쪽으로 절반만
    clean_action = torch.from_numpy(a_pad)[None].cuda().to(torch.bfloat16)

    s_pad = np.zeros((1, 64), dtype=np.float32)
    if args.state_mode == "true":
        s_pad[0, :6] = (anchor - STATE_MEAN) / STATE_STD
    elif args.state_mode == "mean":
        s_pad[0, :6] = 0.0     # 정규화 공간에서 0 = 학습 평균 자세
    # zero 는 그대로 0 (mean 과 같지만 의미를 구분해 남긴다)
    state = torch.from_numpy(s_pad)[None].cuda().to(torch.bfloat16)

    tok = AutoTokenizer.from_pretrained(str(Path(args.base) / "google" / "umt5-xxl"))
    def enc(s):
        o = tok(s, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        return o.input_ids.cuda(), o.attention_mask.cuda()
    ti, tm = enc(args.prompt)
    ni, nm = enc("")

    # ★T5 는 CPU 에 있다. 프롬프트 임베딩을 한 번만 뽑아 GPU 로 옮기고 T5 를 버린다.
    #   (프롬프트가 고정이면 매 블록 재계산할 이유가 없다 — 메모리도 11GB 아낀다)
    with torch.no_grad():
        emb_pos = head.encode_prompt(ti.cpu(), tm.cpu()).cuda()
        emb_neg = head.encode_prompt(ni.cpu(), nm.cpu()).cuda()
    head.encode_prompt = lambda ids, mask: (emb_pos if ids is ti else emb_neg)
    del head.text_encoder
    head.text_encoder = torch.nn.Identity()   # set_frozen_modules_to_eval_mode 가 .eval() 을 부른다
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("  프롬프트 임베딩 %s · T5 해제 후 GPU %.1fGB"
          % (tuple(emb_pos.shape), torch.cuda.memory_allocated() / 2**30))

    print("  이미지 %s -> %s · 액션 %s -> %s · |z|max %.2f"
          % (img.shape, im.shape, acts.shape, tuple(clean_action.shape[1:]), np.abs(norm).max()))

    data = BatchFeature(data=dict(
        images=images, state=state, embodiment_id=torch.zeros(1, dtype=torch.long).cuda(),
        text=ti, text_attention_mask=tm, text_negative=ni, text_attention_mask_negative=nm))

    print("\n=== 3. 생성 (액션 고정, 비디오만 디노이징) ===")
    restore = patch_action_forcing(head, clean_action)
    head.current_start_frame = 0
    head.language = None
    acc = None            # 누적 latent [b, c, t, h, w]
    try:
        with torch.no_grad():
            for b in range(args.blocks):
                t1 = time.time()
                # ★블록 체이닝: videos.shape[2] == 1 이면 모델이 "새 시퀀스" 로 보고 KV 캐시를
                #   리셋한다. 2번째 블록부터는 2프레임을 넘겨 리셋을 피하고, 누적 latent 를
                #   latent_video 로 준다(코드가 [b,c,t,h,w] 를 기대한다 — 전치하면 안 된다).
                d = data
                if b > 0:
                    d = BatchFeature(data={**data, "images": images.repeat(1, 2, 1, 1, 1)})
                out = head.lazy_joint_video_action(None, d, latent_video=acc)
                new_lat = out["video_pred"]                       # [b, c, t, h, w]
                acc = new_lat if acc is None else torch.cat([acc, new_lat], dim=2)
                torch.cuda.empty_cache()      # 블록마다 KV 캐시가 커진다 — 조각모음
                print("  블록 %d/%d · 신규 %s · 누적 %s · start_frame %d · %.1f초 · GPU %.1fGB"
                      % (b + 1, args.blocks, tuple(new_lat.shape[2:3]), tuple(acc.shape),
                         head.current_start_frame, time.time() - t1,
                         torch.cuda.memory_allocated() / 2**30), flush=True)
    finally:
        restore()

    print("\n=== 4. 디코딩 ===")
    with torch.no_grad():
        video = head.vae.decode(acc, tiled=False)
    v = video[0].float().permute(1, 2, 3, 0).clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
    print("  프레임 %s -> 대회 규격 16프레임으로 자름" % (v.shape,))
    v = v[:16]
    import imageio
    imageio.mimwrite(args.out, list(v), fps=6, quality=9)
    print("  -> %s" % args.out)


if __name__ == "__main__":
    main()
