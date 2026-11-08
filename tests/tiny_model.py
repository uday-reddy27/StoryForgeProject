"""Build a tiny, randomly-initialised GPT-2 (same architecture, ~100 KB) with no download.

Used by the test-suite so tests run in seconds and offline. The text it writes is gibberish -
it only proves the *pipeline* works. You can also use it to try the web UI offline:

    python tests/tiny_model.py /tmp/tiny-gpt2
    STORYFORGE_MODEL=/tmp/tiny-gpt2 python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

CORPUS = [
    "The old tower stood silent above the dark forest. A young boy heard screams in the night.",
    "She opened the door and the wind blew the candle out. \"Who is there?\" she whispered.",
    "The detective looked at the clock. It had stopped at midnight, and nobody knew why.",
    "Dragons flew over the mountains while the king counted his gold. The ship sailed on.",
] * 5


def build_tiny_model(out_dir, n_positions: int = 128) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|endoftext|>", bos_token="<|endoftext|>"
    )
    fast.save_pretrained(out)

    eos = fast.eos_token_id
    cfg = GPT2Config(
        vocab_size=len(fast), n_positions=n_positions, n_embd=32, n_layer=2, n_head=2,
        bos_token_id=eos, eos_token_id=eos,
    )
    torch.manual_seed(0)
    GPT2LMHeadModel(cfg).save_pretrained(out)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tiny-gpt2"
    print("Tiny model written to", build_tiny_model(target))
