"""The GPT-2 wrapper: loading, prompt preparation, generation and streaming.

This is the production version of the notebook's ``generate_text`` function.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .params import GenerationParams
from .presets import GENRES, primer_for
from .text_utils import tidy_story

log = logging.getLogger(__name__)

MIN_PROMPT_BUDGET = 16  # always keep at least this many tokens of context for the prompt


class ModelLoadError(RuntimeError):
    """The model/tokenizer could not be loaded (no internet, bad name, out of memory...)."""


@dataclass
class GenerationResult:
    prompt: str
    genre: Optional[str]
    continuations: List[str]   # cleaned text, one per variation
    raw: List[str]             # exactly what the model produced
    params: GenerationParams   # includes the seed that was actually used
    model_name: str
    duration_ms: int


class _StopOnFlag(StoppingCriteria):
    """Lets us abort model.generate() from another thread (e.g. user pressed Stop)."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],), self.event.is_set(), dtype=torch.bool, device=input_ids.device
        )


class StoryGenerator:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_error: Optional[str] = None
        self._load_lock = threading.Lock()  # only one thread loads the model
        self._gen_lock = threading.Lock()   # one generation at a time (a CPU/GPU is the bottleneck)

    # ------------------------------------------------------------------ loading
    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load tokenizer + model once. Safe to call from many threads."""
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            try:
                log.info("Loading model %r on %s ...", self.model_name, self.device)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
            except Exception as exc:  # network, disk, memory, bad name...
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise ModelLoadError(f"Could not load model '{self.model_name}': {exc}") from exc
            # GPT-2 has no padding token; reuse end-of-text (same as the notebook).
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            model.to(self.device).eval()  # eval() switches dropout off
            self.tokenizer = tokenizer
            self.model = model            # set last: `loaded` implies the tokenizer exists
            self.load_error = None
            log.info("Model ready.")

    def load_async(self) -> threading.Thread:
        """Warm the model up in the background so the web server starts instantly."""

        def _run():
            try:
                self.load()
            except ModelLoadError:
                log.exception("Background model load failed")

        thread = threading.Thread(target=_run, name="model-loader", daemon=True)
        thread.start()
        return thread

    @property
    def display_name(self) -> str:
        """Short name for the UI/database: 'gpt2', or just the folder name for a local path."""
        path = Path(self.model_name)
        return path.name if path.exists() else self.model_name

    def info(self) -> dict:
        return {
            "model": self.display_name,
            "device": self.device,
            "loaded": self.loaded,
            "load_error": self.load_error,
        }

    @property
    def n_ctx(self) -> int:
        """Context window: the most tokens (prompt + new) the model can attend to."""
        cfg = self.model.config
        return int(getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", 1024))

    # ------------------------------------------------------------------ preparation
    @staticmethod
    def _check(prompt: str, genre: Optional[str]) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Please enter a story prompt.")
        if genre and genre not in GENRES:
            raise ValueError(f"Unknown genre '{genre}'.")
        return prompt

    def _encode(self, prompt: str, genre: Optional[str], max_new_tokens: int):
        """Turn text into tensors, keeping prompt + new tokens inside the context window.

        GPT-2 can only look at 1024 tokens. If the prompt is too long (e.g. a story that
        has been continued many times) we keep the *end* of it - the most recent part is
        what the next sentence depends on.
        """
        max_new = min(max_new_tokens, self.n_ctx - MIN_PROMPT_BUDGET)
        budget = self.n_ctx - max_new
        enc = self.tokenizer(primer_for(genre) + prompt, return_tensors="pt")
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        if input_ids.shape[1] > budget:
            input_ids, attention_mask = input_ids[:, -budget:], attention_mask[:, -budget:]
        return input_ids.to(self.device), attention_mask.to(self.device), max_new

    def _gen_kwargs(self, params: GenerationParams, input_ids, attention_mask, max_new: int, n_return: int) -> dict:
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,          # NOTE: the notebook used max_length, which counts the prompt too
            do_sample=True,                  # sample instead of always taking the single likeliest token
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            num_return_sequences=n_return,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    # ------------------------------------------------------------------ public API
    def generate(
        self, prompt: str, genre: Optional[str] = None, params: Optional[GenerationParams] = None
    ) -> GenerationResult:
        """Generate one or more continuations of ``prompt`` and return them all at once."""
        prompt = self._check(prompt, genre)
        params = (params or GenerationParams()).with_seed()
        self.load()
        input_ids, attention_mask, max_new = self._encode(prompt, genre, params.max_new_tokens)

        started = time.perf_counter()
        with self._gen_lock:
            torch.manual_seed(params.seed)
            with torch.inference_mode():  # no gradients: faster and uses less memory
                out = self.model.generate(
                    **self._gen_kwargs(params, input_ids, attention_mask, max_new, params.variations)
                )
        # Decode only the NEW tokens - the prompt (and hidden genre primer) are cut off.
        new_tokens = out[:, input_ids.shape[1]:]
        raw = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]
        return GenerationResult(
            prompt=prompt,
            genre=genre or None,
            continuations=[tidy_story(r) for r in raw],
            raw=raw,
            params=params,
            model_name=self.display_name,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    def stream(
        self,
        prompt: str,
        genre: Optional[str] = None,
        params: Optional[GenerationParams] = None,
        cancel: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Yield the continuation piece by piece while the model is still writing.

        model.generate() blocks until it is finished, so it runs in a helper thread and
        pushes text into a queue (TextIteratorStreamer); this generator reads the queue.
        Closing the generator (or setting ``cancel``) stops generation early.
        Streaming supports one variation at a time.
        """
        prompt = self._check(prompt, genre)
        params = (params or GenerationParams()).with_seed()
        self.load()
        input_ids, attention_mask, max_new = self._encode(prompt, genre, params.max_new_tokens)

        cancel = cancel or threading.Event()
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
        )
        kwargs = self._gen_kwargs(params, input_ids, attention_mask, max_new, 1)
        kwargs["streamer"] = streamer
        kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnFlag(cancel)])
        errors: list[BaseException] = []

        def _run():
            try:
                with torch.inference_mode():  # inference_mode is per-thread, so set it here
                    self.model.generate(**kwargs)
            except BaseException as exc:      # make sure the reader never waits forever
                errors.append(exc)
                streamer.end()

        with self._gen_lock:
            torch.manual_seed(params.seed)
            worker = threading.Thread(target=_run, name="generate", daemon=True)
            worker.start()
            try:
                for chunk in streamer:
                    if chunk:  # the streamer holds back partial words and emits "" meanwhile
                        yield chunk
            finally:
                cancel.set()   # tells the model to stop if we are leaving early
                worker.join()
        if errors:
            raise errors[0]
