"""Tests of the real generation pipeline, using the tiny offline GPT-2."""

import os
import threading

import pytest

from storyforge.generator import StoryGenerator
from storyforge.params import GenerationParams

PROMPT = "The old tower stood silent."


def test_generate_returns_requested_variations(generator):
    res = generator.generate(PROMPT, params=GenerationParams(max_new_tokens=20, variations=3, seed=1))
    assert len(res.continuations) == len(res.raw) == 3
    assert res.params.seed == 1 and res.duration_ms >= 0


def test_same_seed_gives_same_story_and_different_seed_differs(generator):
    p = lambda s: GenerationParams(max_new_tokens=25, seed=s)  # noqa: E731
    a = generator.generate(PROMPT, params=p(5)).raw
    b = generator.generate(PROMPT, params=p(5)).raw
    c = generator.generate(PROMPT, params=p(6)).raw
    assert a == b
    assert a != c


def test_missing_seed_is_filled_in_so_story_is_reproducible(generator):
    res = generator.generate(PROMPT, params=GenerationParams(max_new_tokens=20))
    again = generator.generate(PROMPT, params=GenerationParams(max_new_tokens=20, seed=res.params.seed))
    assert res.raw == again.raw


def test_output_excludes_prompt_and_genre_primer(generator):
    res = generator.generate("Zzzz", genre="horror", params=GenerationParams(max_new_tokens=15, seed=3))
    assert "Horror story" not in res.raw[0]
    assert not res.raw[0].startswith("Zzzz")


def test_long_prompt_is_truncated_to_fit_context(generator):
    ids, mask, max_new = generator._encode("word " * 500, "fantasy", 40)
    assert ids.shape[1] + max_new <= generator.n_ctx
    assert ids.shape == mask.shape
    # still generates without a "sequence too long" error
    generator.generate("word " * 500, params=GenerationParams(max_new_tokens=40, seed=1))


def test_oversized_max_new_tokens_is_clamped(generator):
    _, _, max_new = generator._encode("hi", None, 10_000)
    assert max_new <= generator.n_ctx - 16


def test_blank_prompt_and_unknown_genre_are_rejected(generator):
    with pytest.raises(ValueError):
        generator.generate("   ")
    with pytest.raises(ValueError):
        generator.generate(PROMPT, genre="western")


def test_stream_matches_non_streaming_output(generator):
    params = GenerationParams(max_new_tokens=30, seed=11)
    chunks = list(generator.stream(PROMPT, params=params))
    assert len(chunks) >= 1
    assert "".join(chunks) == generator.generate(PROMPT, params=params).raw[0]


def test_stream_can_be_cancelled_and_releases_the_lock(generator):
    cancel = threading.Event()
    stream = generator.stream(PROMPT, params=GenerationParams(max_new_tokens=60, seed=2), cancel=cancel)
    next(stream)
    stream.close()  # what happens when the browser disconnects
    # If the lock leaked or the worker thread hung, this would deadlock:
    assert generator.generate(PROMPT, params=GenerationParams(max_new_tokens=10, seed=2)).raw


def test_bad_model_name_raises_clear_error(tmp_path):
    from storyforge.generator import ModelLoadError

    gen = StoryGenerator(str(tmp_path / "does-not-exist"), device="cpu")
    with pytest.raises(ModelLoadError):
        gen.generate(PROMPT)
    assert gen.load_error and not gen.loaded


@pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL") != "1",
                    reason="downloads real GPT-2 (~500 MB); run with RUN_REAL_MODEL=1")
def test_real_gpt2_smoke():
    gen = StoryGenerator("gpt2")
    res = gen.generate("The detective opened the door and", params=GenerationParams(max_new_tokens=40, seed=1))
    assert len(res.continuations[0]) > 20
