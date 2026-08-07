"""V100 이 왜 느린지 — 원가속(matmul)과 어텐션 커널을 따로 잰다.

5090 과 같은 스크립트를 돌려 비교한다. 원가속 차이는 3~5배가 정상이고,
그보다 훨씬 크게 벌어지면 어텐션 백엔드(flash/mem-efficient 미지원 → math 폴백)가 범인이다.
"""
import time

import torch
import torch.nn.functional as F
from torch.backends.cuda import SDPBackend, sdp_kernel  # noqa: F401  (구버전 호환용)

dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| capability", torch.cuda.get_device_capability(0))


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


# ---------- ① 원가속 ----------
n = 4096
a = torch.randn(n, n, device=dev, dtype=torch.float16)
b = torch.randn(n, n, device=dev, dtype=torch.float16)
dt = timeit(lambda: a @ b)
print(f"\n① fp16 matmul {n}^3 : {dt*1000:.2f} ms → {2*n**3/dt/1e12:.1f} TFLOPS")

a32, b32 = a.float(), b.float()
dt32 = timeit(lambda: a32 @ b32)
print(f"   fp32 matmul      : {dt32*1000:.2f} ms → {2*n**3/dt32/1e12:.1f} TFLOPS")

# ---------- ② SDPA 백엔드 ----------
# 생성 때 실제로 쓰이는 모양: 공간 어텐션 (b*t, heads, 2560, 64)
B, H, S, D = 16, 5, 2560, 64
q = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.float16)

print(f"\n② SDPA (b={B}, heads={H}, seq={S}, dim={D}) — 공간 어텐션과 같은 모양")
try:
    from torch.nn.attention import SDPBackend as NB, sdpa_kernel
    backends = [("flash", NB.FLASH_ATTENTION), ("mem_efficient", NB.EFFICIENT_ATTENTION), ("math", NB.MATH)]
    for name, be in backends:
        try:
            with sdpa_kernel(be):
                dt = timeit(lambda: F.scaled_dot_product_attention(q, k, v), warmup=2, iters=5)
            print(f"   {name:14s} {dt*1000:8.1f} ms")
        except Exception as e:
            print(f"   {name:14s} 사용 불가 — {type(e).__name__}: {str(e)[:70]}")
except ImportError:
    dt = timeit(lambda: F.scaled_dot_product_attention(q, k, v), warmup=2, iters=5)
    print(f"   기본 백엔드     {dt*1000:8.1f} ms")

dt = timeit(lambda: F.scaled_dot_product_attention(q, k, v), warmup=2, iters=5)
print(f"   자동 선택      {dt*1000:8.1f} ms")

# ---------- ③ 이 저장소가 실제로 쓰는 어텐션 ----------
print("\n③ 저장소 어텐션 구현")
try:
    import xformers  # noqa: F401
    print("   xformers 있음")
except ImportError:
    print("   xformers 없음 → 저장소가 naive einsum softmax 로 폴백할 수 있다")

# 순수 einsum 방식 (lvdm 의 폴백 경로)
def naive():
    sim = torch.einsum("bhid,bhjd->bhij", q, k) * (D ** -0.5)
    return torch.einsum("bhij,bhjd->bhid", sim.softmax(dim=-1), v)


try:
    dt = timeit(naive, warmup=1, iters=3)
    print(f"   naive einsum   {dt*1000:8.1f} ms")
except RuntimeError as e:
    print(f"   naive einsum   OOM/실패 — {str(e)[:80]}")

print(f"\n최대 메모리 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
