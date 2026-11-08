# StoryForge - AI Story Generator (GPT-2 + Flask)

Write the first line of a story and GPT-2 finishes it. Built from the YBI notebook
*"Text Generation using transformer models"* and grown into a complete, tested web application:
streaming output, genre styles, sampling controls, reproducible seeds, saved history, a REST API and a CLI.

```
Browser (HTML/CSS/JS)  --JSON / server-sent events-->  Flask API  -->  StoryGenerator (GPT-2, PyTorch)
                                                           |
                                                      SQLite history
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                        # then open http://127.0.0.1:5000
```

The first start downloads GPT-2 (~500 MB, cached afterwards). The page shows a status badge
while the model loads in the background. No GPU is needed; generation speed depends on your CPU (expect several seconds to well under a minute for a 200-token story); the streaming view shows text as it is written.

**Command line** (closest to the original notebook):

```bash
python -m storyforge.cli "A young boy hears screams from a tower." --genre horror --tokens 200
python -m storyforge.cli            # asks "Enter a story prompt:" like the notebook
```

**Tests** (offline, ~1 second - they use a tiny randomly-initialised GPT-2, so no download):

```bash
pytest
RUN_REAL_MODEL=1 pytest tests/test_generator.py::test_real_gpt2_smoke   # optional: real GPT-2
```

**Try the UI without downloading anything** (the text will be gibberish, but everything else is real):

```bash
python tests/tiny_model.py tiny-gpt2
STORYFORGE_MODEL=tiny-gpt2 python app.py
```

## Free online demo: the browser version (`web/`)

`web/` is a second, static version of the app in which GPT-2 runs **inside the visitor's browser** using
[Transformers.js](https://huggingface.co/docs/transformers.js). With no Python server it can be hosted for free (for
example on Vercel). It shares the idea and the UI of the Python version but has no seed/variations, and saves history in
the browser instead of SQLite. Deployment steps for GitHub + Vercel are in **[DEPLOY.md](DEPLOY.md)**.

```bash
cd web && python -m http.server 8000      # then open http://localhost:8000
node --test web/tests/text-utils.test.mjs # unit tests for the JavaScript helpers
```

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `STORYFORGE_MODEL` | `gpt2` | Any causal LM: `distilgpt2` (faster), `gpt2-medium` (better, slower) or a local folder |
| `STORYFORGE_DB` | `instance/storyforge.db` | SQLite file for saved stories |
| `STORYFORGE_MAX_NEW_TOKENS` | `400` | Server-side cap on story length |
| `STORYFORGE_MAX_PROMPT_CHARS` | `4000` | Max prompt size |
| `STORYFORGE_PRELOAD` | `1` | `0` = load the model on first request instead of at start-up |
| `HOST`, `PORT` | `127.0.0.1`, `5000` | Where the server listens |

## REST API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Model name, device, `loaded`, `load_error` |
| `GET /api/presets` | Genres with example prompts |
| `POST /api/generate` | Returns all variations at once |
| `POST /api/generate/stream` | Server-sent events: `start`, many `token`, then `done` (or `error`) |
| `GET /api/stories`, `GET/DELETE /api/stories/<id>` | Saved history |

Request body for both generate endpoints:

```json
{ "prompt": "The old tower stood silent.", "genre": "horror",
  "params": { "max_new_tokens": 200, "temperature": 0.8, "top_k": 50, "top_p": 0.92,
              "repetition_penalty": 1.15, "no_repeat_ngram_size": 4, "variations": 1, "seed": 42 } }
```

```bash
curl -X POST localhost:5000/api/generate -H 'Content-Type: application/json' \
     -d '{"prompt":"The old tower stood silent.","params":{"max_new_tokens":60}}'
```

Invalid input returns `400 {"error": "..."}` with a readable message; a model that failed to load returns `503`.

## How it works

1. **Tokenizer** - GPT-2 doesn't read letters or words; it reads *tokens* (sub-word pieces, ~50,000 of them).
   `"The old tower"` becomes a list of integers.
2. **Language model** - GPT-2 is a 12-layer Transformer (124 M parameters). Given the tokens so far, it outputs a
   probability for every possible next token. It has no concept of "write a story"; it only continues text.
3. **Autoregressive loop** - pick one token, append it, feed everything back in, repeat. `max_new_tokens` is how
   many times that loop runs.
4. **Sampling** decides *which* token to pick. Always taking the most likely one gives bland, looping text, so we sample:
   - `temperature` - reshapes the probabilities. Low = predictable, high = adventurous (and eventually nonsense).
   - `top_k` - only the k likeliest tokens are candidates.
   - `top_p` - only the smallest group of tokens whose probabilities add up to p is a candidate.
   - `repetition_penalty` / `no_repeat_ngram_size` - fight GPT-2's habit of repeating itself.
   - `seed` - fixes the random number generator, so the same inputs reproduce the same story.
5. **Genre styles** - GPT-2 can't follow instructions, so a genre is a hidden "book header" such as
   `"Horror story\n\nChapter One\n\n"` placed before your prompt (see `presets.py`). The model sees what looks like the
   start of a horror chapter and continues in kind. The header is never shown.
6. **Context window** - GPT-2 sees at most 1024 tokens. If a prompt is longer (for example after pressing *Continue*
   several times) the *end* is kept, because the next sentence depends most on the latest text.
7. **Streaming** - `model.generate()` blocks until finished, so it runs in a helper thread and pushes text into a
   queue (`TextIteratorStreamer`). Flask reads the queue and forwards each piece as a server-sent event, which gives the
   typing effect. Pressing *Stop* (or closing the tab) sets a flag that a `StoppingCriteria` checks every step.
8. **Clean-up** - trims the dangling half-sentence at the end, removes broken UTF-8 characters, collapses blank lines.

## What changed from the original notebook

| Notebook | StoryForge | Why |
|---|---|---|
| `max_length=250` | `max_new_tokens` | `max_length` counts the prompt too, so a long prompt silently left no room for the story |
| `repetition_penalty=1.5` | `1.15` + `no_repeat_ngram_size=4` | 1.5 is so harsh the model avoids ordinary words - the notebook's sample output turns incoherent ("...Jigsaw :)") |
| Model loaded at import time | Loaded once, in the background, thread-safe | Fast start-up; the page shows model status |
| `input()` prompt | Web UI, REST API and CLI | Usable by people and by other programs |
| Random every run | Seed returned with every story | Reproducible results |
| Prompt + story printed together | Only new tokens decoded; prompt shown separately | Cleaner output, hidden genre primer |
| Ends mid-sentence | Trimmed to the last full sentence | Reads like a finished passage |
| Nothing saved | SQLite history | Revisit, continue or delete stories |
| No checks | Validated, range-limited inputs; one generation at a time | A bad request can't crash or hog the server |
| No tests | 41 automated tests (+1 optional real-model test) | Safe to change |

## Project structure

```
app.py                  entry point:  python app.py
storyforge/
  config.py             settings from environment variables
  params.py             GenerationParams: defaults + validation
  presets.py            genre primers and example prompts
  text_utils.py         tidy_story, finish_last_sentence, join_story
  generator.py          StoryGenerator: load / generate / stream  (the core)
  storage.py            SQLite history
  web.py                Flask app factory, REST + SSE endpoints
  cli.py                command-line interface
templates/index.html    the page
static/app.js           streaming client, history, controls (vanilla JS, no build step)
static/style.css        light/dark theme, responsive
tests/                  unit, pipeline and API tests (+ tiny_model.py helper)
web/                    browser-only version (static files, deploys free on Vercel)
DEPLOY.md               GitHub + Vercel deployment guide
notebooks/              the original YBI notebook, untouched
```

## Limitations (honest)

- GPT-2 (2019, 124 M parameters) is small: stories are fun and atmospheric but drift, contradict themselves and
  forget characters. It does not follow instructions. A bigger model (`gpt2-large`, or an instruction-tuned model) would
  need only a change of `STORYFORGE_MODEL` for the basic flow, though the genre-primer trick would then be unnecessary.
- The default sampling values are sensible starting points chosen from how these parameters behave, not tuned on a
  benchmark - adjust them in *Advanced settings*.
- Generation is serialised (one story at a time per server process), which suits a single-user demo. Serving many users
  needs a queue or a GPU service.
- Generated text is not fact-checked or moderated; GPT-2 can produce odd or offensive text.

## Ideas to extend

- React front-end (the API is already decoupled) or user accounts with Django/MySQL.
- Fine-tune GPT-2 on your own corpus with the Hugging Face `Trainer` for a signature style.
- Swap in an instruction-tuned model and add real "write me a story about..." prompts.
- Export a story to PDF/Markdown; share links; rate stories and use ratings to tune defaults.
- Docker + gunicorn deployment.

## Interview talking points

- Built an end-to-end NLP application around a pre-trained Transformer (GPT-2) with Hugging Face and PyTorch: a Flask
  REST API, token-by-token streaming over server-sent events, SQLite persistence and a responsive vanilla-JS front-end.
- Explain the decoding parameters (temperature, top-k, top-p, repetition penalty) and why sampling beats greedy decoding.
- Handled real engineering concerns: thread-safe lazy model loading, request validation, cancelling a running generation,
  context-window truncation, reproducible seeds, and an offline test suite built on a tiny model.
