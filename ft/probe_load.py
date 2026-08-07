"""UNet 을 실제로 만들어 backbone.ckpt 를 실어보고, 몇 개가 실리는지 센다.

add_act_time_emb True/False 두 설정으로 각각 돌려 차이를 확인한다.
대회 config 원본 값(model_channels 32 등)도 함께 돌려 대조군으로 쓴다.

GPU 불필요 — meta device 를 쓰지 않고 CPU 에 만들되, 로드는 shape 대조만 하므로 몇 분.
"""
import sys, os
from collections import OrderedDict
import torch

KIT = os.path.expanduser("~/ft/kit/baseline/challenge_kit")
sys.path.insert(0, os.path.join(KIT, "libs/dynamicrafter"))
sys.path.insert(0, os.path.join(KIT, "src"))
sys.path.insert(0, os.path.expanduser("~/ft/kit/baseline/shared_libs/video_utils"))

from lvdm.modules.networks.openaimodel3d import UNetModel  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/ft/checkpoints/backbone.ckpt")

print(f"적재 중: {CKPT}")
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
for k in ("state_dict", "module"):
    if isinstance(raw, dict) and k in raw:
        raw = raw[k]
        break
PFX = "model.diffusion_model."
ref = OrderedDict((k[len(PFX):], v) for k, v in raw.items() if k.startswith(PFX))
print(f"UNet 텐서 {len(ref)}개\n")

# ---------- ordered channel_mult ----------
mc = ref["time_embed.0.weight"].shape[1]
seq = []
for i in range(64):
    key = f"input_blocks.{i}.0.out_layers.3.weight"
    if key in ref:
        seq.append(ref[key].shape[0] // mc)
# num_res_blocks 개씩 묶어 레벨 하나 — 같은 배수가 연속해도 레벨은 따로 센다
NUM_RES = 2
ordered = [seq[i] for i in range(0, len(seq), NUM_RES)]
print(f"input_blocks 배수 순서: {seq}")
print(f"→ channel_mult = {ordered}\n")

# DynamiCrafter 표준 구성 + 대회 config 의 액션 관련 항목
BASE = dict(
    in_channels=8, out_channels=4,
    model_channels=mc,
    attention_resolutions=[4, 2, 1],
    num_res_blocks=2,
    channel_mult=ordered,
    num_head_channels=64,
    transformer_depth=1,
    context_dim=1024,
    use_linear=True,
    use_checkpoint=True,
    temporal_conv=True,
    temporal_attention=True,
    temporal_selfatt_only=True,
    use_relative_position=False,
    use_causal_attention=False,
    temporal_length=16,
    addition_attention=True,
    image_cross_attention=True,
    fs_condition=False,
    default_fs=10,
    use_scale_shift_norm=True,
    dropout=0.1,
)

CASES = [
    ("사전학습 구조 + 액션조건 OFF (상한선)", dict(BASE, action_conditioned=False)),
    ("액션조건 ON · add_act_time_emb=True", dict(BASE, action_conditioned=True, action_dims=6, add_act_time_emb=True)),
    ("액션조건 ON · add_act_time_emb=False (현재 기본값)", dict(BASE, action_conditioned=True, action_dims=6, add_act_time_emb=False)),
    ("액션조건 ON · True · 12차원(C안)", dict(BASE, action_conditioned=True, action_dims=12, add_act_time_emb=True)),
    ("대회 원본 config (11M)", dict(
        in_channels=8, out_channels=4, model_channels=32,
        attention_resolutions=[4, 2], num_res_blocks=2, channel_mult=[1, 2, 3],
        num_head_channels=16, transformer_depth=1, context_dim=1024,
        use_linear=True, use_checkpoint=True, temporal_conv=True,
        temporal_attention=True, temporal_selfatt_only=True,
        use_relative_position=False, use_causal_attention=False,
        temporal_length=16, addition_attention=True, image_cross_attention=True,
        use_scale_shift_norm=True, dropout=0.1,
        action_conditioned=True, action_dims=6,
    )),
]

print("=" * 88)
print(f"{'구성':<46} {'전체':>7} {'로드':>7} {'비율':>7} {'랜덤':>7}")
print("=" * 88)
for name, params in CASES:
    try:
        with torch.device("meta"):
            m = UNetModel(**params)
    except Exception as e:
        print(f"{name:<46}  생성 실패: {type(e).__name__}: {e}")
        continue
    own = dict(m.state_dict())
    total = len(own)
    hit = sum(1 for k, v in own.items() if k in ref and tuple(ref[k].shape) == tuple(v.shape))
    print(f"{name:<46} {total:>7} {hit:>7} {hit/total*100:>6.1f}% {total-hit:>7}")

    if "add_act_time_emb" in params:
        miss_key = [k for k, v in own.items()
                    if k.startswith("time_embed") and (k not in ref or tuple(ref[k].shape) != tuple(v.shape))]
        if miss_key:
            print(f"{'':<46}   └ 못 싣는 time_embed 텐서: {miss_key}")
print("=" * 88)
