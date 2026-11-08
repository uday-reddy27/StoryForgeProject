"""Command-line version - the closest thing to the original notebook.

    python -m storyforge.cli "A young boy hears screams from a tower." --genre horror
    python -m storyforge.cli            # asks for a prompt, like the notebook's input()
"""

from __future__ import annotations

import argparse
import os
import sys

from .generator import ModelLoadError, StoryGenerator
from .params import GenerationParams
from .presets import GENRES
from .text_utils import join_story


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a story continuation with GPT-2.")
    p.add_argument("prompt", nargs="?", help="Opening lines of the story (asked interactively if omitted)")
    p.add_argument("--genre", choices=sorted(GENRES), help="Optional genre style")
    p.add_argument("--tokens", type=int, default=200, help="Tokens to generate (default 200, ~150 words)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--variations", type=int, default=1)
    p.add_argument("--seed", type=int, help="Reuse a seed to reproduce a story")
    p.add_argument("--model", default=os.environ.get("STORYFORGE_MODEL", "gpt2"))
    args = p.parse_args(argv)

    prompt = args.prompt or input("Enter a story prompt: ")
    try:
        params = GenerationParams.from_dict({
            "max_new_tokens": args.tokens, "temperature": args.temperature, "top_k": args.top_k,
            "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
            "variations": args.variations, "seed": args.seed,
        })
        result = StoryGenerator(args.model).generate(prompt, args.genre, params)
    except (ValueError, ModelLoadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for i, text in enumerate(result.continuations, 1):
        if len(result.continuations) > 1:
            print(f"\n--- Variation {i} ---")
        print("\nGenerated Text:\n")
        print(join_story(result.prompt, text))
    print(f"\n[model={result.model_name} seed={result.params.seed} time={result.duration_ms/1000:.1f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
