"""모델 수프 — 두 갈래의 **정점 가중치를 평균**내 하나로 만든다.

**왜 될 만한가.** mw 0(base-10)과 mw 2.0 은 base-09 누적 4,200 이라는 **같은 조상**에서 갈라졌고,
구조가 동일하며(motion_weight 는 손실에만 작용), 각자의 정점 점수가 0.2306 / 0.2299 로 거의 같다.
같은 지점에서 출발해 비슷한 높이에 도착한 두 해는 **선형 경로로 이어져 있을 가능성**이 높고,
그럴 때 가중치 평균이 두 해보다 낮은 지점에 떨어지는 일이 있다.

실패해도 잃는 것은 생성 시간뿐이다.
"""
import sys, torch

a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

A = torch.load(a, map_location="cpu"); A = A.get("state_dict", A)
B = torch.load(b, map_location="cpu"); B = B.get("state_dict", B)
ka, kb = set(A), set(B)
print("키 A %d · B %d · 공통 %d · A만 %d · B만 %d" % (len(ka), len(kb), len(ka & kb), len(ka - kb), len(kb - ka)))
for k in sorted(ka - kb)[:5]: print("  A만:", k)
for k in sorted(kb - ka)[:5]: print("  B만:", k)

soup, skip = {}, 0
for k in sorted(ka & kb):
    x, y = A[k], B[k]
    if not torch.is_tensor(x) or x.shape != y.shape or not x.is_floating_point():
        soup[k] = x; skip += 1; continue
    soup[k] = (x.float() * (1 - w) + y.float() * w).to(x.dtype)
print("평균 %d개 · 그대로 %d개 · 비율 %.2f:%.2f" % (len(soup) - skip, skip, 1 - w, w))
torch.save(soup, out)
print("저장:", out)
