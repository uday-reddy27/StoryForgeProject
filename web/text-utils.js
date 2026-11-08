// Model-free text helpers - a JavaScript port of storyforge/text_utils.py

const TERMINAL_AT_END = /[.!?]["'”’)\]]*$/;
const SENTENCE_END = /[.!?]["'”’)\]]*(?=\s|$)/g;

/** Cut a story back to its last complete sentence (only if that keeps >= minKeep of the text). */
export function finishLastSentence(text, minKeep = 0.4) {
  text = text.replace(/\s+$/, "");
  if (!text || TERMINAL_AT_END.test(text)) return text;
  let cut = -1;
  let m;
  SENTENCE_END.lastIndex = 0;
  while ((m = SENTENCE_END.exec(text)) !== null) cut = m.index + m[0].length;
  if (cut === -1) return text;
  return cut >= text.length * minKeep ? text.slice(0, cut) : text;
}

/** Clean raw model output for display. */
export function tidyStory(text, finishSentence = true) {
  text = text.replace(/�/g, "");          // half a UTF-8 character left by a token cut
  text = text.replace(/[ \t]+\n/g, "\n");      // trailing spaces on lines
  text = text.replace(/\n{3,}/g, "\n\n");      // at most one blank line
  text = text.replace(/\s+$/, "");
  return finishSentence ? finishLastSentence(text) : text;
}

/** prompt + continuation, adding a space only when the model's text needs one. */
export function joinStory(prompt, continuation) {
  if (continuation && !/^\s/.test(continuation) && /[.!?"”]$/.test(prompt)) {
    return prompt + " " + continuation;
  }
  return prompt + continuation;
}
