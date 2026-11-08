"""Small, model-free text helpers (easy to unit-test)."""

from __future__ import annotations

import re

_TERMINAL_AT_END = re.compile(r"[.!?][\"'”’)\]]*$")
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s|$)")


def finish_last_sentence(text: str, min_keep: float = 0.4) -> str:
    """Cut a story back to its last complete sentence.

    A model that stops after N tokens usually stops mid-sentence ("...and then the door").
    We trim to the last sentence-ending mark - but only if that keeps at least
    ``min_keep`` of the text, so a long unpunctuated passage is not butchered.
    """
    text = text.rstrip()
    if not text or _TERMINAL_AT_END.search(text):
        return text
    ends = list(_SENTENCE_END.finditer(text))
    if not ends:
        return text
    cut = ends[-1].end()
    return text[:cut] if cut >= len(text) * min_keep else text


def tidy_story(text: str, finish_sentence: bool = True) -> str:
    """Clean raw model output for display."""
    # GPT-2 works on bytes; a token cut can leave half a UTF-8 character at the end.
    text = text.replace("�", "")
    text = re.sub(r"[ \t]+\n", "\n", text)       # trailing spaces on lines
    text = re.sub(r"\n{3,}", "\n\n", text)       # never more than one blank line
    text = text.rstrip()
    if finish_sentence:
        text = finish_last_sentence(text)
    return text


def join_story(prompt: str, continuation: str) -> str:
    """prompt + continuation, adding a space only when the model's text needs one."""
    if continuation and not continuation[0].isspace() and prompt[-1:] in ".!?\"”":
        return prompt + " " + continuation
    return prompt + continuation
