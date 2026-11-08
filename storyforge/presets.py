"""Genre presets.

GPT-2 is *not* instruction-tuned: it cannot be told "write a horror story". It only
continues text. So a genre is implemented as a short hidden "primer" placed in front of
the user's prompt (like a book header). The model sees text that looks like the start of
a horror chapter and continues in that style. The primer is never shown in the output.
"""

from __future__ import annotations

GENRES = {
    "fantasy": {
        "label": "Fantasy",
        "primer": "Fantasy story\n\nChapter One\n\n",
        "example": "The last dragon egg began to glow beneath the ruins of the old watchtower.",
    },
    "scifi": {
        "label": "Sci-Fi",
        "primer": "Science fiction story\n\nChapter One\n\n",
        "example": "The ship's navigator woke to find that every star in the sky had moved overnight.",
    },
    "horror": {
        "label": "Horror",
        "primer": "Horror story\n\nChapter One\n\n",
        "example": "A young boy hears screams from a tower and sees torch lights flashing in the dark.",
    },
    "mystery": {
        "label": "Mystery",
        "primer": "Detective mystery story\n\nChapter One\n\n",
        "example": "The detective noticed that the clock in the study had stopped at exactly midnight.",
    },
    "romance": {
        "label": "Romance",
        "primer": "Romance story\n\nChapter One\n\n",
        "example": "She had promised herself she would never return to the seaside town, until the letter arrived.",
    },
    "adventure": {
        "label": "Adventure",
        "primer": "Adventure story\n\nChapter One\n\n",
        "example": "The map was torn in half, and the only person who owned the other piece was a pirate.",
    },
}


def primer_for(genre: str | None) -> str:
    """Hidden text placed before the prompt. Empty string when no genre is chosen."""
    if genre and genre in GENRES:
        return GENRES[genre]["primer"]
    return ""


def public_presets() -> list[dict]:
    """What the UI needs: id, label and an example prompt (never the primer)."""
    return [
        {"id": gid, "label": g["label"], "example": g["example"]}
        for gid, g in GENRES.items()
    ]
