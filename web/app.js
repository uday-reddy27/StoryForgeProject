/* StoryForge - browser-only version.
 *
 * GPT-2 runs inside the visitor's browser through Transformers.js (ONNX Runtime Web / WebAssembly),
 * so the site is just static files and can be hosted for free (Vercel, GitHub Pages, Netlify...).
 *
 * Flow: Generate -> load model (first time only, cached by the browser) -> stream tokens into a card
 *       -> tidy the text -> save to localStorage.
 */
import {
  pipeline, env, TextStreamer, InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { tidyStory, joinStory } from "./text-utils.js";
import { GENRES, primerFor } from "./presets.js";

env.allowLocalModels = false;   // always fetch models from the Hugging Face Hub (then cached)

const CONTEXT = 1024;           // GPT-2 can attend to at most 1024 tokens
const HISTORY_KEY = "storyforge.history.v1";
const HISTORY_MAX = 30;

const $ = (id) => document.getElementById(id);
const el = {
  prompt: $("prompt"), count: $("count"), generate: $("generate"), stop: $("stop"),
  error: $("error"), stories: $("stories"), empty: $("empty"),
  history: $("history"), historyEmpty: $("history-empty"), clearHistory: $("clear-history"),
  status: $("status"), statusText: $("status-text"),
  length: $("length"), lengthOut: $("length-out"), temp: $("temperature"), tempOut: $("temp-out"),
  model: $("model"), example: $("example-btn"), genres: $("genres"),
  loadbar: $("loadbar"), loadProgress: $("load-progress"), loadText: $("load-text"),
};
const genreLabel = Object.fromEntries(GENRES.map((g) => [g.id, g.label]));

let stopper = null;             // InterruptableStoppingCriteria of the running generation
let stoppedByUser = false;
let busy = false;

// ------------------------------------------------------------------ small helpers
const selectedGenre = () => document.querySelector('input[name="genre"]:checked').value;
const showError = (msg) => { el.error.textContent = msg; el.error.hidden = !msg; };
const shortModel = (id) => id.replace("Xenova/", "");

function setStatus(state, text) {
  el.status.dataset.state = state;
  el.statusText.textContent = text;
}

function tempLabel(t) {
  return t < 0.6 ? "focused" : t < 0.95 ? "balanced" : t < 1.15 ? "imaginative" : "wild";
}
function updateLabels() {
  const tokens = Number(el.length.value);
  el.lengthOut.textContent = `${tokens} tokens (≈ ${Math.round(tokens * 0.75)} words)`;
  const t = Number(el.temp.value);
  el.tempOut.textContent = `${t.toFixed(2)} · ${tempLabel(t)}`;
  el.count.textContent = `${el.prompt.value.length} / ${el.prompt.maxLength}`;
}

/** Read a number input, check its range, and say which field is wrong. */
function readNumber(id, label, min, max, isInt = false) {
  const input = $(id);
  const v = Number(input.value);
  if (input.value.trim() === "" || Number.isNaN(v) || v < min || v > max || (isInt && !Number.isInteger(v))) {
    throw new Error(`${label} must be ${isInt ? "a whole number " : ""}between ${min} and ${max}.`);
  }
  return v;
}

function readSettings() {
  return {
    max_new_tokens: Number(el.length.value),
    temperature: Number(el.temp.value),
    top_k: readNumber("top_k", "Top-k", 1, 200, true),
    top_p: readNumber("top_p", "Top-p", 0.1, 1),
    repetition_penalty: readNumber("repetition_penalty", "Repetition penalty", 1, 2),
    no_repeat_ngram_size: readNumber("no_repeat_ngram_size", "No-repeat n-gram", 0, 10, true),
  };
}

function setBusy(on) {
  busy = on;
  el.generate.disabled = on;
  el.stop.hidden = !on;
  el.generate.textContent = on ? "Writing…" : "Generate story";
}

// ------------------------------------------------------------------ model loading
const pipelines = new Map();    // model id -> Promise of the loaded pipeline

/** Load a model once; progress_callback reports every file, so we add the files up for one bar. */
function loadModel(modelId) {
  if (pipelines.has(modelId)) return pipelines.get(modelId);

  // Only keep one model in memory: release the others.
  for (const [id, p] of pipelines) {
    p.then((g) => g.dispose && g.dispose()).catch(() => {});
    pipelines.delete(id);
  }

  const files = new Map();      // file name -> { loaded, total }
  const onProgress = (info) => {
    if (info.status === "progress" && info.total) {
      files.set(info.file, { loaded: info.loaded, total: info.total });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded; total += f.total; }
      const pct = total ? Math.min(100, (loaded / total) * 100) : 0;
      el.loadProgress.value = pct;
      el.loadText.textContent = `Downloading model… ${pct.toFixed(0)}% (${(loaded / 1e6).toFixed(0)} MB)`;
      setStatus("loading", `Downloading ${shortModel(modelId)}… ${pct.toFixed(0)}%`);
    }
  };

  const promise = pipeline("text-generation", modelId, {
    dtype: "q8",                // 8-bit quantised weights: about 4x smaller than full precision
    device: "wasm",
    progress_callback: onProgress,
  });
  pipelines.set(modelId, promise);
  promise.catch(() => pipelines.delete(modelId));   // allow a retry after a failure
  return promise;
}

// ------------------------------------------------------------------ story cards
// IMPORTANT: user/model text is always inserted with textContent, never innerHTML (prevents XSS).
function makeCard({ label, genre, prompt, continuation = "", streaming = false }) {
  const card = document.createElement("article");
  card.className = "story-card" + (streaming ? " streaming" : "");

  const head = document.createElement("div");
  head.className = "story-head";
  const left = document.createElement("span");
  left.textContent = label;
  head.append(left);
  if (genre) {
    const b = document.createElement("span");
    b.className = "badge"; b.textContent = genreLabel[genre] || genre;
    head.append(b);
  }

  const text = document.createElement("p");
  text.className = "story-text";
  const p = document.createElement("span"); p.className = "prompt-part"; p.textContent = prompt;
  const g = document.createElement("span"); g.className = "gen-part"; g.textContent = continuation;
  text.append(p, g);

  const actions = document.createElement("div");
  actions.className = "story-actions";
  card.append(head, text, actions);
  el.empty.hidden = true;
  el.stories.prepend(card);
  return { card, gen: g, actions };
}

function linkButton(label, onClick) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "link"; b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

function finishCard(parts, { story, continuation, metaText }) {
  parts.card.classList.remove("streaming");
  parts.gen.textContent = continuation;
  parts.actions.replaceChildren();
  const copy = linkButton("Copy", async (e) => {
    try { await navigator.clipboard.writeText(story); e.target.textContent = "Copied ✓"; }
    catch { e.target.textContent = "Copy failed"; }
    setTimeout(() => (e.target.textContent = "Copy"), 1500);
  });
  const meta = document.createElement("span");
  meta.className = "meta"; meta.textContent = metaText;
  parts.actions.append(linkButton("Continue ▸", () => continueStory(story)), copy, meta);

  if (!continuation.trim()) addNote(parts, "The model stopped right away. Try again, or raise the length / creativity.");
}

function addNote(parts, message) {
  const n = document.createElement("p");
  n.className = "note"; n.textContent = message;
  parts.card.append(n);
}

// ------------------------------------------------------------------ generation
async function generate() {
  if (busy) return;
  showError("");
  const prompt = el.prompt.value.trim();
  if (!prompt) { showError("Please write a story prompt first."); el.prompt.focus(); return; }

  let settings;
  try { settings = readSettings(); } catch (err) { showError(err.message); return; }

  const genre = selectedGenre() || null;
  const modelId = el.model.value;
  setBusy(true);
  const parts = makeCard({ label: "Story", genre, prompt, streaming: true });
  let phase = "model";            // tells us where an error happened
  stoppedByUser = false;

  try {
    // 1) model (downloads the first time, instant afterwards)
    const first = !pipelines.has(modelId);
    if (first) { el.loadbar.hidden = false; el.loadProgress.value = 0; el.loadText.textContent = "Preparing model…"; setStatus("loading", `Loading ${shortModel(modelId)}…`); }
    const generator = await loadModel(modelId);
    el.loadbar.hidden = true;
    setStatus("ready", `${shortModel(modelId)} ready · in your browser`);
    phase = "generate";

    // 2) prepare the input: hidden genre header + prompt, trimmed to fit the 1024-token window
    const tokenizer = generator.tokenizer;
    const maxNew = Math.min(settings.max_new_tokens, CONTEXT - 16);
    let input = primerFor(genre) + prompt;
    const ids = tokenizer.encode(input);
    const budget = CONTEXT - maxNew;
    if (ids.length > budget) input = tokenizer.decode(ids.slice(-budget), { skip_special_tokens: true });

    // 3) generate, streaming text into the card as it is produced
    stopper = new InterruptableStoppingCriteria();
    const parts_ = [];
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (piece) => { if (piece) { parts_.push(piece); parts.gen.textContent += piece; } },
    });
    const started = performance.now();
    await generator(input, {
      ...settings,
      max_new_tokens: maxNew,
      do_sample: true,
      return_full_text: false,
      streamer,
      stopping_criteria: [stopper],
    });
    const seconds = (performance.now() - started) / 1000;

    if (stoppedByUser) {
      parts.card.classList.remove("streaming");
      addNote(parts, "Stopped. (Partial text is not saved.)");
      return;
    }

    // 4) clean up, show, save
    const continuation = tidyStory(parts_.join(""));
    const story = joinStory(prompt, continuation);
    finishCard(parts, { story, continuation, metaText: `${shortModel(modelId)} · ${seconds.toFixed(1)}s` });
    saveStory({ prompt, genre, continuation, model: modelId, params: settings, seconds });
    renderHistory();
  } catch (err) {
    parts.card.remove();
    if (!el.stories.querySelector(".story-card")) el.empty.hidden = false;
    el.loadbar.hidden = true;
    const msg = (err && err.message) || String(err);
    if (phase === "model") {
      setStatus("error", "Model failed to load");
      showError(`The model could not be downloaded (${msg}). Check your internet connection and try again, or choose DistilGPT-2 under Advanced settings.`);
    } else {
      setStatus("error", "Generation failed");
      showError(`Generation failed: ${msg}`);
    }
  } finally {
    stopper = null;
    setBusy(false);
  }
}

function continueStory(story) {
  el.prompt.value = story.slice(-el.prompt.maxLength);
  updateLabels();
  generate();
}

// ------------------------------------------------------------------ history (localStorage)
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); } catch { return []; }
}
function writeHistory(items) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, HISTORY_MAX))); } catch { /* private mode / quota */ }
}
function saveStory(s) {
  const items = loadHistory();
  items.unshift({ id: Date.now(), created_at: new Date().toISOString(), ...s });
  writeHistory(items);
}

function renderHistory() {
  const items = loadHistory();
  el.history.replaceChildren();
  el.historyEmpty.hidden = items.length > 0;
  el.clearHistory.hidden = items.length === 0;
  for (const s of items) el.history.append(historyItem(s));
}

function historyItem(s) {
  const li = document.createElement("li");
  li.className = "history-item";

  const open = document.createElement("button");
  open.type = "button"; open.className = "open";
  const title = document.createElement("span"); title.className = "title"; title.textContent = s.prompt;
  const sub = document.createElement("span"); sub.className = "sub";
  const when = new Date(s.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
  sub.textContent = `${s.genre ? (genreLabel[s.genre] || s.genre) + " · " : ""}${when}`;
  open.append(title, sub);
  open.addEventListener("click", () => {
    const parts = makeCard({ label: "Saved story", genre: s.genre, prompt: s.prompt, continuation: s.continuation });
    finishCard(parts, {
      story: joinStory(s.prompt, s.continuation), continuation: s.continuation,
      metaText: `${shortModel(s.model || "")} · ${Number(s.seconds || 0).toFixed(1)}s`,
    });
    parts.card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  const del = document.createElement("button");
  del.type = "button"; del.className = "del"; del.title = "Delete"; del.setAttribute("aria-label", "Delete story");
  del.textContent = "×";
  del.addEventListener("click", () => { writeHistory(loadHistory().filter((x) => x.id !== s.id)); renderHistory(); });

  li.append(open, del);
  return li;
}

// ------------------------------------------------------------------ wiring
for (const g of GENRES) {   // genre chips
  const label = document.createElement("label");
  label.className = "chip";
  const input = document.createElement("input");
  input.type = "radio"; input.name = "genre"; input.value = g.id;
  const span = document.createElement("span"); span.textContent = g.label;
  label.append(input, span);
  el.genres.append(label);
}

el.generate.addEventListener("click", generate);
el.stop.addEventListener("click", () => {
  if (stopper) { stoppedByUser = true; stopper.interrupt(); }
});
el.clearHistory.addEventListener("click", () => { writeHistory([]); renderHistory(); });
el.prompt.addEventListener("input", updateLabels);
el.length.addEventListener("input", updateLabels);
el.temp.addEventListener("input", updateLabels);
el.prompt.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
});
el.example.addEventListener("click", () => {
  let g = GENRES.find((x) => x.id === selectedGenre());
  if (!g) {
    g = GENRES[Math.floor(Math.random() * GENRES.length)];
    document.querySelector(`input[name="genre"][value="${g.id}"]`).checked = true;
  }
  el.prompt.value = g.example;
  updateLabels();
  el.prompt.focus();
});

updateLabels();
renderHistory();
