// Run with:  node --test web/tests/text-utils.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { finishLastSentence, tidyStory, joinStory } from "../text-utils.js";
import { GENRES, primerFor } from "../presets.js";

test("trims a dangling fragment", () => {
  assert.equal(
    finishLastSentence("The door creaked open. A cold wind rushed in. Then the lights"),
    "The door creaked open. A cold wind rushed in.");
});

test("keeps complete text, including closing quotes", () => {
  assert.equal(finishLastSentence('He said, "Run!"'), 'He said, "Run!"');
});

test("does not butcher long unpunctuated text", () => {
  const text = "Hi. " + "and then ".repeat(30);
  assert.equal(finishLastSentence(text), text.trimEnd());
});

test("ignores decimal points", () => {
  const t = "It cost 3.5 dollars and then";
  assert.equal(finishLastSentence(t), t);
});

test("tidyStory cleans artifacts", () => {
  assert.equal(tidyStory("Line one.   \n\n\n\nLine two�", false), "Line one.\n\nLine two");
});

test("joinStory spacing", () => {
  assert.equal(joinStory("Hello world", " and more"), "Hello world and more");
  assert.equal(joinStory("It was dark.", "He ran."), "It was dark. He ran.");
  assert.equal(joinStory("It was dark.", "\nHe ran."), "It was dark.\nHe ran.");
  assert.equal(joinStory("Prompt", ""), "Prompt");
});

test("genres", () => {
  assert.equal(primerFor(""), "");
  assert.equal(primerFor("nope"), "");
  assert.ok(primerFor("horror").startsWith("Horror story"));
  assert.equal(new Set(GENRES.map((g) => g.id)).size, GENRES.length);
});
