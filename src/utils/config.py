"""Config 로더 — base + exp + server 3계층 딥머지 (동작).

설계(docs/07_운영전략 §4, docs/13_팀_룰북 §1):
  - 실험 1개 = configs/exp/<expNN_이름>.yaml 1개 (+ 커밋 해시 1개). 단일 변수 변경 원칙.
  - 하드웨어 종속값(batch/precision/경로)만 configs/server/<이름>.yaml로 분리 → 실험 config는 서버 무관.
병합 우선순위(뒤가 우선): base < exp < server < CLI override.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
BASE_CONFIG = CONFIG_ROOT / "base.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    """중첩 dict 재귀 병합(over 우선). list/scalar는 통째 교체(부분 병합 안 함)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path | str | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config 없음: {p}")
    import yaml  # 지연 import — utils import를 가볍게

    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """CLI dotted override 적용. 예: ["train.lr=1e-4", "data.stride=4"]."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override 형식 오류(key=value 필요): {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return out


def _coerce(s: str) -> Any:
    """override 값 문자열 → int/float/bool/None/str 추정."""
    low = s.strip().lower()
    if low in ("null", "none"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def load_config(exp: str | None = None, server: str | None = None,
                base: str | Path = BASE_CONFIG, overrides: list[str] | None = None) -> dict:
    """base → exp → server → CLI override 순으로 병합한 최종 config(dict) 반환.

    실험별 config는 서버 무관이어야 하므로 exp에는 하드웨어 종속값을 넣지 않는다(server가 담당).
    """
    cfg = _load_yaml(base)
    cfg = _deep_merge(cfg, _load_yaml(exp))
    cfg = _deep_merge(cfg, _load_yaml(server))
    cfg = _apply_overrides(cfg, overrides)
    cfg.setdefault("_meta", {})
    cfg["_meta"].update({"exp": str(exp) if exp else None, "server": str(server) if server else None})
    return cfg
