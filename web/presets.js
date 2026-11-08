// Genre presets - same idea as storyforge/presets.py.
// GPT-2 can't follow instructions, so a genre is a hidden "book header" placed before the prompt.

export const GENRES = [
  { id: "fantasy", label: "Fantasy", primer: "Fantasy story\n\nChapter One\n\n",
    example: "The last dragon egg began to glow beneath the ruins of the old watchtower." },
  { id: "scifi", label: "Sci-Fi", primer: "Science fiction story\n\nChapter One\n\n",
    example: "The ship's navigator woke to find that every star in the sky had moved overnight." },
  { id: "horror", label: "Horror", primer: "Horror story\n\nChapter One\n\n",
    example: "A young boy hears screams from a tower and sees torch lights flashing in the dark." },
  { id: "mystery", label: "Mystery", primer: "Detective mystery story\n\nChapter One\n\n",
    example: "The detective noticed that the clock in the study had stopped at exactly midnight." },
  { id: "romance", label: "Romance", primer: "Romance story\n\nChapter One\n\n",
    example: "She had promised herself she would never return to the seaside town, until the letter arrived." },
  { id: "adventure", label: "Adventure", primer: "Adventure story\n\nChapter One\n\n",
    example: "The map was torn in half, and the only person who owned the other piece was a pirate." },
];

export function primerFor(id) {
  const g = GENRES.find((x) => x.id === id);
  return g ? g.primer : "";
}
