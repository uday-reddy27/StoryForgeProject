"""Model-free unit tests: text helpers, parameter validation, storage."""

import pytest

from storyforge.params import GenerationParams
from storyforge.storage import StoryStore
from storyforge.text_utils import finish_last_sentence, join_story, tidy_story


# ----------------------------------------------------------------- text helpers
def test_finish_last_sentence_trims_dangling_fragment():
    text = "The door creaked open. A cold wind rushed in. Then the lights"
    assert finish_last_sentence(text) == "The door creaked open. A cold wind rushed in."


def test_finish_last_sentence_keeps_complete_text():
    assert finish_last_sentence('He said, "Run!"') == 'He said, "Run!"'


def test_finish_last_sentence_does_not_butcher_long_unpunctuated_text():
    text = "Hi. " + "and then " * 30  # the only full sentence is a tiny part
    assert finish_last_sentence(text) == text.rstrip()


def test_finish_last_sentence_ignores_decimal_points():
    text = "It cost 3.5 dollars and then"
    assert finish_last_sentence(text) == text


def test_tidy_story_cleans_artifacts():
    raw = "Line one.   \n\n\n\nLine two�"
    assert tidy_story(raw, finish_sentence=False) == "Line one.\n\nLine two"


def test_join_story_spacing():
    assert join_story("Hello world", " and more") == "Hello world and more"
    assert join_story("It was dark.", "He ran.") == "It was dark. He ran."
    assert join_story("It was dark.", "\nHe ran.") == "It was dark.\nHe ran."
    assert join_story("Prompt", "") == "Prompt"


# ----------------------------------------------------------------- parameters
def test_params_defaults_are_valid():
    p = GenerationParams.from_dict({})
    assert p.max_new_tokens == 200 and p.variations == 1 and p.seed is None


def test_params_accepts_numeric_strings():
    p = GenerationParams.from_dict({"temperature": "0.5", "max_new_tokens": "120", "seed": "42"})
    assert (p.temperature, p.max_new_tokens, p.seed) == (0.5, 120, 42)


@pytest.mark.parametrize("bad", [
    {"temperature": 0}, {"temperature": 9}, {"temperature": "hot"}, {"top_p": 0},
    {"max_new_tokens": 5}, {"max_new_tokens": 10**9}, {"max_new_tokens": 12.5},
    {"variations": 4}, {"repetition_penalty": 0.5}, {"seed": -1}, {"top_k": True},
    {"temperature": float("nan")},
])
def test_params_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        GenerationParams.from_dict(bad)


def test_params_respects_server_cap():
    with pytest.raises(ValueError):
        GenerationParams.from_dict({"max_new_tokens": 300}, max_new_tokens_cap=100)


def test_with_seed_fills_missing_seed_and_keeps_given_one():
    assert GenerationParams().with_seed().seed is not None
    assert GenerationParams(seed=7).with_seed().seed == 7


# ----------------------------------------------------------------- storage
def test_store_roundtrip(tmp_path):
    store = StoryStore(tmp_path / "s.db")
    sid = store.add(prompt="Once upon", genre="fantasy", continuation=" a time.",
                    params={"seed": 1}, model="gpt2", duration_ms=12)
    got = store.get(sid)
    assert got["story"] == "Once upon a time."
    assert got["params"] == {"seed": 1}
    assert [s["id"] for s in store.list()] == [sid]
    assert store.delete(sid) is True
    assert store.get(sid) is None and store.delete(sid) is False
