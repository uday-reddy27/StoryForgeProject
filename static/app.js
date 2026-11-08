/* StoryForge front-end - plain JavaScript, no build step.
 *
 * Flow:  Generate click -> POST /api/generate/stream -> read server-sent events
 *        -> append tokens to a story card as they arrive -> refresh the history list.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const cfg = window.STORYFORGE;

  const el = {
    prompt: $("prompt"), count: $("count"), generate: $("generate"), stop: $("stop"),
    error: $("error"), stories: $("stories"), empty: $("empty"), history: $("history"),
    historyEmpty: $("history-empty"), status: $("status"), statusText: $("status-text"),
    length: $("length"), lengthOut: $("length-out"), temp: $("temperature"), tempOut: $("temp-out"),
    example: $("example-btn"),
  };
  const genreLabel = Object.fromEntries(cfg.genres.map((g) => [g.id, g.label]));
  let controller = null; // AbortController of the running request

  // ---------------------------------------------------------------- small helpers
  const selectedGenre = () => document.querySelector('input[name="genre"]:checked').value;
  const num = (id) => $(id).value.trim();

  function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }

  function tempLabel(t) {
    return t < 0.6 ? "focused" : t < 0.95 ? "balanced" : t < 1.15 ? "imaginative" : "wild";
  }
  function updateLabels() {
    const tokens = Number(el.length.value);
    el.lengthOut.textContent = `${tokens} tokens (≈ ${Math.round(tokens * 0.75)} words)`;
    const t = Number(el.temp.value);
    el.tempOut.textContent = `${t.toFixed(2)} · ${tempLabel(t)}`;
    el.count.textContent = `${el.prompt.value.length} / ${cfg.maxPromptChars}`;
  }

  function buildPayload() {
    const params = {
      max_new_tokens: Number(el.length.value),
      temperature: Number(el.temp.value),
      top_k: num("top_k"), top_p: num("top_p"),
      repetition_penalty: num("repetition_penalty"),
      no_repeat_ngram_size: num("no_repeat_ngram_size"),
      variations: num("variations"),
      seed: num("seed"),
    };
    return { prompt: el.prompt.value, genre: selectedGenre() || null, params };
  }

  function setBusy(busy) {
    el.generate.disabled = busy;
    el.stop.hidden = !busy;
    el.generate.textContent = busy ? "Writing…" : "Generate story";
  }

  // ---------------------------------------------------------------- story cards
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
    return { card, gen: g, actions, text };
  }

  function finishCard(parts, { id, story, continuation, seed, duration_ms }) {
    parts.card.classList.remove("streaming");
    parts.gen.textContent = continuation;           // replace raw stream with the tidied text
    parts.actions.replaceChildren();

    const cont = button("Continue ▸", () => continueStory(story));
    const copy = button("Copy", async (e) => {
      try { await navigator.clipboard.writeText(story); e.target.textContent = "Copied ✓"; }
      catch { e.target.textContent = "Copy failed"; }
      setTimeout(() => (e.target.textContent = "Copy"), 1500);
    });
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `#${id} · seed ${seed} · ${(duration_ms / 1000).toFixed(1)}s`;
    parts.actions.append(cont, copy, meta);

    if (!continuation.trim()) {
      const n = document.createElement("p");
      n.className = "note";
      n.textContent = "The model stopped right away. Try again, or raise the length / creativity.";
      parts.card.append(n);
    }
  }

  function button(label, onClick) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "link"; b.textContent = label;
    b.addEventListener("click", onClick);
    return b;
  }

  function addCard(parts) {
    el.empty.hidden = true;
    el.stories.prepend(parts.card);
  }

  // ---------------------------------------------------------------- generation
  async function generate() {
    showError("");
    const payload = buildPayload();
    if (!payload.prompt.trim()) { showError("Please write a story prompt first."); el.prompt.focus(); return; }

    controller = new AbortController();
    setBusy(true);
    const promptText = payload.prompt.trim();
    const multi = Number(payload.params.variations) > 1;
    try {
      if (multi) await generateBatch(payload, promptText);
      else await generateStreaming(payload, promptText);
      await loadHistory();
    } catch (err) {
      if (err.name !== "AbortError") showError(err.message || "Something went wrong.");
    } finally {
      setBusy(false);
      controller = null;
    }
  }

  async function generateBatch(payload, promptText) {
    const data = await postJson("/api/generate", payload);
    data.results.forEach((r, i) => {
      const parts = makeCard({ label: `Variation ${i + 1}`, genre: payload.genre, prompt: promptText });
      addCard(parts);
      finishCard(parts, { ...r, seed: data.seed, duration_ms: data.duration_ms });
    });
  }

  async function generateStreaming(payload, promptText) {
    const res = await fetch("/api/generate/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), signal: controller.signal,
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `Request failed (${res.status})`);

    const parts = makeCard({ label: "Story", genre: payload.genre, prompt: promptText, streaming: true });
    addCard(parts);
    let seed = null, finished = false;

    try {
      for await (const evt of readEvents(res)) {
        if (evt.type === "start") {
          seed = evt.seed;
          if (evt.warming_up) parts.gen.textContent = "";
        } else if (evt.type === "token") {
          parts.gen.textContent += evt.text;
        } else if (evt.type === "done") {
          finished = true;
          finishCard(parts, { ...evt, seed });
        } else if (evt.type === "error") {
          throw new Error(evt.message);
        }
      }
    } catch (err) {
      if (!finished) {
        parts.card.classList.remove("streaming");
        if (err.name === "AbortError") addStoppedNote(parts);
        else parts.card.remove();
      }
      throw err;
    }
  }

  function addStoppedNote(parts) {
    const n = document.createElement("p");
    n.className = "note";
    n.textContent = "Stopped. (Partial text is not saved.)";
    parts.card.append(n);
  }

  // Parse a text/event-stream response body: events are "data: {json}\n\n".
  async function* readEvents(res) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (block.startsWith("data: ")) yield JSON.parse(block.slice(6));
      }
    }
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller && controller.signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  function continueStory(story) {
    el.prompt.value = story;
    updateLabels();
    generate();
  }

  // ---------------------------------------------------------------- history
  async function loadHistory() {
    try {
      const { stories } = await (await fetch("/api/stories?limit=20")).json();
      el.history.replaceChildren();
      el.historyEmpty.hidden = stories.length > 0;
      for (const s of stories) el.history.append(historyItem(s));
    } catch { /* history is a nicety; ignore network hiccups */ }
  }

  function historyItem(s) {
    const li = document.createElement("li");
    li.className = "history-item";

    const open = document.createElement("button");
    open.type = "button"; open.className = "open";
    const title = document.createElement("span"); title.className = "title"; title.textContent = s.prompt;
    const sub = document.createElement("span"); sub.className = "sub";
    const when = new Date(s.created_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    sub.textContent = `${s.genre ? genreLabel[s.genre] + " · " : ""}${when}`;
    open.append(title, sub);
    open.addEventListener("click", () => {
      const parts = makeCard({ label: `Saved #${s.id}`, genre: s.genre, prompt: s.prompt, continuation: s.continuation });
      addCard(parts);
      finishCard(parts, { id: s.id, story: s.story, continuation: s.continuation, seed: s.params.seed, duration_ms: s.duration_ms });
      parts.card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    const del = document.createElement("button");
    del.type = "button"; del.className = "del"; del.title = "Delete"; del.setAttribute("aria-label", "Delete story");
    del.textContent = "×";
    del.addEventListener("click", async () => {
      await fetch(`/api/stories/${s.id}`, { method: "DELETE" });
      loadHistory();
    });

    li.append(open, del);
    return li;
  }

  // ---------------------------------------------------------------- model status
  async function pollHealth() {
    let again = true;
    try {
      const h = await (await fetch("/api/health")).json();
      if (h.loaded) {
        el.status.dataset.state = "ready";
        el.statusText.textContent = `${h.model} ready · ${h.device}`;
        again = false;
      } else if (h.load_error) {
        el.status.dataset.state = "error";
        el.statusText.textContent = "Model failed to load";
        el.status.title = h.load_error;
        showError(`The model could not be loaded: ${h.load_error}`);
        again = false;
      } else {
        el.status.dataset.state = "loading";
        el.statusText.textContent = `Loading ${h.model}… (first run downloads ~500 MB)`;
      }
    } catch {
      el.status.dataset.state = "error";
      el.statusText.textContent = "Server not reachable";
    }
    if (again) setTimeout(pollHealth, 2000);
  }

  // ---------------------------------------------------------------- wiring
  el.generate.addEventListener("click", generate);
  el.stop.addEventListener("click", () => controller && controller.abort());
  el.prompt.addEventListener("input", updateLabels);
  el.length.addEventListener("input", updateLabels);
  el.temp.addEventListener("input", updateLabels);
  el.prompt.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !el.generate.disabled) generate();
  });
  el.example.addEventListener("click", () => {
    let g = cfg.genres.find((x) => x.id === selectedGenre());
    if (!g) {
      g = cfg.genres[Math.floor(Math.random() * cfg.genres.length)];
      document.querySelector(`input[name="genre"][value="${g.id}"]`).checked = true;
    }
    el.prompt.value = g.example;
    updateLabels();
    el.prompt.focus();
  });

  updateLabels();
  pollHealth();
  loadHistory();
})();
