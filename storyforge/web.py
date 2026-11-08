"""Flask application: REST API + server-sent-events streaming + the web page."""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import HTTPException

from .config import ROOT, Settings
from .generator import ModelLoadError, StoryGenerator
from .params import GenerationParams
from .presets import GENRES, public_presets
from .storage import StoryStore
from .text_utils import join_story, tidy_story


class ApiError(Exception):
    """An error we want to show the user as {"error": "..."} with a proper HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def create_app(
    settings: Optional[Settings] = None,
    generator: Optional[StoryGenerator] = None,
    store: Optional[StoryStore] = None,
) -> Flask:
    """Application factory. Passing generator/store in makes the app easy to test."""
    settings = settings or Settings.from_env()
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # reject giant request bodies

    injected = generator is not None
    generator = generator or StoryGenerator(settings.model_name)
    store = store or StoryStore(settings.db_path)
    if settings.preload_model and not injected:
        generator.load_async()

    # ------------------------------------------------------------------ helpers
    def parse_request() -> tuple[str, Optional[str], GenerationParams]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("Request body must be a JSON object.")
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ApiError("Please enter a story prompt.")
        prompt = prompt.strip()
        if len(prompt) > settings.max_prompt_chars:
            raise ApiError(f"Prompt is too long (max {settings.max_prompt_chars} characters).")
        genre = data.get("genre") or None
        if genre is not None and genre not in GENRES:
            raise ApiError(f"Unknown genre '{genre}'.")
        raw_params = data.get("params") or {}
        if not isinstance(raw_params, dict):
            raise ApiError("'params' must be an object.")
        try:
            params = GenerationParams.from_dict(raw_params, settings.max_new_tokens_cap)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return prompt, genre, params

    def sse(payload: dict) -> str:
        """Format one server-sent event."""
        return f"data: {json.dumps(payload)}\n\n"

    # ------------------------------------------------------------------ error handling
    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return jsonify(error=exc.message), exc.status

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(error=exc.description), exc.code
        return exc

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        app.logger.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return jsonify(error="Something went wrong on the server. Check the server log."), 500
        return "Internal Server Error", 500

    # ------------------------------------------------------------------ pages & meta
    @app.get("/")
    def index():
        return render_template(
            "index.html",
            genres=public_presets(),
            max_tokens=settings.max_new_tokens_cap,
            max_prompt_chars=settings.max_prompt_chars,
        )

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", **generator.info())

    @app.get("/api/presets")
    def presets():
        return jsonify(genres=public_presets())

    # ------------------------------------------------------------------ generation
    @app.post("/api/generate")
    def generate():
        prompt, genre, params = parse_request()
        try:
            result = generator.generate(prompt, genre, params)
        except ModelLoadError as exc:
            raise ApiError(str(exc), 503) from None
        except ValueError as exc:
            raise ApiError(str(exc)) from None

        results = []
        for text in result.continuations:
            story_id = store.add(
                prompt=prompt, genre=genre, continuation=text, params=result.params.to_dict(),
                model=result.model_name, duration_ms=result.duration_ms,
            )
            results.append({"id": story_id, "continuation": text, "story": join_story(prompt, text)})
        return jsonify(
            results=results, seed=result.params.seed, model=result.model_name,
            duration_ms=result.duration_ms,
        )

    @app.post("/api/generate/stream")
    def generate_stream():
        prompt, genre, params = parse_request()
        params = params.with_seed()
        params = GenerationParams(**{**params.to_dict(), "variations": 1})  # streaming = 1 story
        cancel = threading.Event()

        def events():
            yield sse({"type": "start", "seed": params.seed, "model": generator.display_name,
                       "warming_up": not generator.loaded})
            started = time.perf_counter()
            parts: list[str] = []
            stream = generator.stream(prompt, genre, params, cancel=cancel)
            try:
                for chunk in stream:
                    parts.append(chunk)
                    yield sse({"type": "token", "text": chunk})
            except ModelLoadError as exc:
                yield sse({"type": "error", "message": str(exc)})
                return
            except ValueError as exc:
                yield sse({"type": "error", "message": str(exc)})
                return
            except Exception:
                app.logger.exception("Streaming generation failed")
                yield sse({"type": "error", "message": "Generation failed. Check the server log."})
                return
            finally:
                stream.close()  # if the browser disconnected, this stops the model
                cancel.set()

            text = tidy_story("".join(parts))
            duration = int((time.perf_counter() - started) * 1000)
            story_id = store.add(
                prompt=prompt, genre=genre, continuation=text, params=params.to_dict(),
                model=generator.display_name, duration_ms=duration,
            )
            yield sse({"type": "done", "id": story_id, "continuation": text,
                       "story": join_story(prompt, text), "duration_ms": duration})

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ history
    @app.get("/api/stories")
    def list_stories():
        limit = max(1, min(request.args.get("limit", 30, type=int), 100))
        return jsonify(stories=store.list(limit))

    @app.get("/api/stories/<int:story_id>")
    def get_story(story_id: int):
        story = store.get(story_id)
        if story is None:
            raise ApiError("Story not found.", 404)
        return jsonify(story)

    @app.delete("/api/stories/<int:story_id>")
    def delete_story(story_id: int):
        if not store.delete(story_id):
            raise ApiError("Story not found.", 404)
        return jsonify(deleted=story_id)

    return app
