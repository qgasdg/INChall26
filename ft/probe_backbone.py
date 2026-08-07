"""backbone.ckpt(DynamiCrafter_512) 의 텐서 shape 에서 UNet config 를 역산한다.

목적 두 가지:
  ① 우리 config 의 model_channels · channel_mult · context_dim 을 추측이 아니라 확정한다.
  ② B 가설 판정 — add_act_time_emb=False 면 time_embed 의 마지막 층 출력이 embed_dim//2 가 되어
     사전학습 텐서와 shape 이 어긋난다. 어긋나면 그 층은 로드되지 않고 랜덤 초기화로 남는다.

GPU 불필요. CPU 로 몇 분.
"""
import sys
from collections import OrderedDict
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/backbone.ckpt"

print(f"적재 중: {CKPT}")
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
for k in ("state_dict", "module"):
    if isinstance(sd, dict) and k in sd:
        sd = sd[k]
        break
print(f"텐서 {len(sd)}개\n")

PFX = "model.diffusion_model."
unet = OrderedDict((k[len(PFX):], v) for k, v in sd.items() if k.startswith(PFX))
if not unet:
    unet = OrderedDict((k, v) for k, v in sd.items() if k.startswith("input_blocks"))
    PFX = ""
print(f"UNet 접두사 '{PFX}' → {len(unet)}개\n")

# ---------- ① 구조 역산 ----------
te0 = unet["time_embed.0.weight"]       # [embed_dim, model_channels]
te2 = unet["time_embed.2.weight"]       # [time_embed_dim, embed_dim]
embed_dim, model_channels = te0.shape
time_embed_dim = te2.shape[0]

print("=" * 70)
print("① 구조")
print("=" * 70)
print(f"  time_embed.0.weight  {tuple(te0.shape)}  → model_channels = {model_channels}, embed_dim = {embed_dim}")
print(f"  time_embed.2.weight  {tuple(te2.shape)}  → time_embed_dim = {time_embed_dim}")
print(f"  embed_dim == model_channels*4 ? {embed_dim == model_channels * 4}")

inp0 = unet["input_blocks.0.0.weight"]
print(f"  input_blocks.0.0     {tuple(inp0.shape)}  → in_channels = {inp0.shape[1]}")
out = unet["out.2.weight"]
print(f"  out.2.weight         {tuple(out.shape)}  → out_channels = {out.shape[0]}")

# channel_mult — 각 input_block 의 출력 채널을 model_channels 로 나눈 값
mults, seen = [], set()
for i in range(64):
    key = f"input_blocks.{i}.0.out_layers.3.weight"
    if key not in unet:
        continue
    m = unet[key].shape[0] // model_channels
    if m not in seen:
        seen.add(m)
        mults.append(m)
print(f"  channel_mult (관측)  {mults}")

ctx = [v.shape[1] for k, v in unet.items() if k.endswith("attn2.to_k.weight")]
print(f"  context_dim          {sorted(set(ctx))}")

nres = sum(1 for k in unet if k.startswith("input_blocks.1.0.out_layers.3.weight"))
tl = [tuple(v.shape) for k, v in unet.items() if "relative_position" in k or "temporal" in k][:3]
print(f"  temporal 관련 텐서 예시 {tl}")

# ---------- ② B 가설 판정 ----------
print()
print("=" * 70)
print("② B 가설 — add_act_time_emb")
print("=" * 70)
print(f"  사전학습 time_embed.2.weight : {tuple(te2.shape)}")
print(f"  add_act_time_emb=True  일 때 우리가 만드는 것 : ({embed_dim}, {embed_dim})")
print(f"  add_act_time_emb=False 일 때 우리가 만드는 것 : ({embed_dim // 2}, {embed_dim})")
ok_true = tuple(te2.shape) == (embed_dim, embed_dim)
ok_false = tuple(te2.shape) == (embed_dim // 2, embed_dim)
print()
print(f"  → True  : {'로드됨' if ok_true else '★ shape 불일치 → 로드 안 됨'}")
print(f"  → False : {'로드됨' if ok_false else '★ shape 불일치 → 로드 안 됨 (랜덤 초기화)'}")

# ---------- ③ 액션 관련 텐서 유무 ----------
print()
print("=" * 70)
print("③ 액션 관련 텐서 (있으면 6→12차원 변경 시 잃는 게 생긴다)")
print("=" * 70)
act = [k for k in unet if "action" in k.lower()]
print(f"  {act if act else '없음 — action_embed 는 어차피 처음부터 학습. action_dims 변경 비용 0'}")
