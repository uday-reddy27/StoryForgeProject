"""Runtime settings, read from environment variables so nothing is hard-coded."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Any causal-LM name on the Hugging Face Hub ("gpt2", "distilgpt2", "gpt2-medium")
    # or a local folder saved with save_pretrained().
    model_name: str = "gpt2"
    db_path: Path = ROOT / "instance" / "storyforge.db"
    max_new_tokens_cap: int = 400      # hard server-side limit on story length
    max_prompt_chars: int = 4000       # protects the server from huge inputs
    preload_model: bool = True         # load the model in the background at start-up

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_name=os.environ.get("STORYFORGE_MODEL", cls.model_name),
            db_path=Path(os.environ.get("STORYFORGE_DB", ROOT / "instance" / "storyforge.db")),
            max_new_tokens_cap=int(os.environ.get("STORYFORGE_MAX_NEW_TOKENS", cls.max_new_tokens_cap)),
            max_prompt_chars=int(os.environ.get("STORYFORGE_MAX_PROMPT_CHARS", cls.max_prompt_chars)),
            preload_model=os.environ.get("STORYFORGE_PRELOAD", "1") != "0",
        )
