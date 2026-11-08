"""Generation parameters with validation.

Every number that reaches model.generate() comes through here, so a bad request
(temperature=-5, max_new_tokens=10**9 ...) is rejected with a clear message instead of
crashing the model or tying up the server.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GenerationParams:
    max_new_tokens: int = 200        # length of the *new* text (the prompt is not counted)
    temperature: float = 0.8         # <1 = safer/more predictable, >1 = wilder
    top_k: int = 50                  # sample only from the k most likely next tokens
    top_p: float = 0.92              # ...and only from the smallest set covering p probability
    repetition_penalty: float = 1.15 # >1 discourages re-using tokens already in the text
    no_repeat_ngram_size: int = 4    # forbid repeating any exact 4-token phrase (0 = off)
    variations: int = 1              # how many alternative continuations to produce
    seed: Optional[int] = None       # same seed + same prompt + same settings = same story

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, max_new_tokens_cap: int = 400) -> "GenerationParams":
        data = data or {}
        d = cls()
        seed = data.get("seed")
        return cls(
            max_new_tokens=_coerce(data, "max_new_tokens", int, 10, max_new_tokens_cap, d.max_new_tokens),
            temperature=_coerce(data, "temperature", float, 0.1, 1.5, d.temperature),
            top_k=_coerce(data, "top_k", int, 1, 200, d.top_k),
            top_p=_coerce(data, "top_p", float, 0.1, 1.0, d.top_p),
            repetition_penalty=_coerce(data, "repetition_penalty", float, 1.0, 2.0, d.repetition_penalty),
            no_repeat_ngram_size=_coerce(data, "no_repeat_ngram_size", int, 0, 10, d.no_repeat_ngram_size),
            variations=_coerce(data, "variations", int, 1, 3, d.variations),
            seed=None if seed in (None, "") else _coerce(data, "seed", int, 0, 2**31 - 1, 0),
        )

    # ------------------------------------------------------------------ helpers
    def with_seed(self) -> "GenerationParams":
        """Return a copy that definitely has a seed, so any story can be reproduced."""
        if self.seed is not None:
            return self
        return replace(self, seed=secrets.randbelow(2**31))

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce(data: Mapping[str, Any], key: str, cast, lo, hi, default):
    raw = data.get(key)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        raise ValueError(f"'{key}' must be a number.")
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.") from None
    if cast is int and isinstance(raw, float) and raw != value:
        raise ValueError(f"'{key}' must be a whole number.")
    if not (lo <= value <= hi):  # also rejects NaN
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return value
