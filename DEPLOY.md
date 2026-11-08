# Free deployment: GitHub + Vercel

There are two versions of StoryForge in this repo:

| | Python version (`app.py`) | Browser version (`web/`) |
|---|---|---|
| Where GPT-2 runs | on a server (PyTorch) | **in each visitor's browser** (Transformers.js) |
| Hosting | needs a server with ~2 GB RAM (not free) | **static files - free on Vercel** |
| Seed / variations / saved history on server | yes | no (history is kept in the browser) |
| Best for | local use, API, learning the full stack | a free public demo link |

Vercel cannot run the Python version (PyTorch + the model do not fit its functions), so deploy the **`web/`** folder.

## 1. Test the browser version on your computer first

```
cd web
python -m http.server 8000
```

Open http://localhost:8000 (it must be `http://`, not a double-clicked file), write a prompt and press **Generate**.
The first run downloads the model (roughly 100-150 MB) - the status badge and progress bar show it. Later runs use the
browser's cache.

If GPT-2 fails to download but **DistilGPT-2** (Advanced settings -> Model) works, tell me - the model file name
on the Hub is the one thing I could not check from my environment.

## 2. Push to GitHub

Create an **empty** repository on github.com, then in the project folder:

```
git init
git add .
git status          # .venv and instance must NOT be listed
git commit -m "StoryForge: GPT-2 story generator"
git branch -M main
git remote add origin https://github.com/<your-username>/storyforge.git
git push -u origin main
```

## 3. Deploy on Vercel (free Hobby plan)

1. Sign in at vercel.com with your GitHub account -> **Add New... -> Project** -> import the `storyforge` repo.
2. **Root Directory: click Edit and choose `web`.** This matters: the repo root contains `app.py` (Flask), which Vercel
   would try to build as a Python function and fail.
3. Framework Preset: **Other**. Leave Build Command and Output Directory empty. Click **Deploy**.
4. Open the `*.vercel.app` link. Every `git push` to `main` redeploys automatically.

(Command-line alternative: `cd web` then `npx vercel --prod`.)

## 4. After it is live

- Put the link, a screenshot and a short GIF at the top of `README.md`.
- First visit by a visitor = one model download; their browser caches it afterwards.
- Speed depends on the visitor's device (the model runs on their CPU). Mention that in your portfolio description.
