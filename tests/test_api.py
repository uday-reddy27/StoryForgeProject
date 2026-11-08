"""HTTP-level tests of the Flask app."""

import json


def post(client, url, **body):
    return client.post(url, json=body)


def parse_sse(text):
    return [json.loads(block[len("data: "):]) for block in text.strip().split("\n\n") if block]


def test_index_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "StoryForge" in html and 'id="prompt"' in html and "Horror" in html


def test_health_and_presets(client):
    h = client.get("/api/health").get_json()
    assert h["status"] == "ok" and h["loaded"] is True
    ids = [g["id"] for g in client.get("/api/presets").get_json()["genres"]]
    assert "fantasy" in ids and "horror" in ids
    assert "primer" not in json.dumps(client.get("/api/presets").get_json())  # internals stay private


def test_generate_saves_story_and_history_endpoints_work(client):
    r = post(client, "/api/generate", prompt="The old tower", genre="horror",
             params={"max_new_tokens": 20, "seed": 9, "variations": 2})
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["results"]) == 2 and body["seed"] == 9
    first = body["results"][0]
    assert first["story"].startswith("The old tower")

    listed = client.get("/api/stories").get_json()["stories"]
    assert len(listed) == 2 and listed[0]["genre"] == "horror"

    one = client.get(f"/api/stories/{first['id']}").get_json()
    assert one["prompt"] == "The old tower" and one["params"]["seed"] == 9

    assert client.delete(f"/api/stories/{first['id']}").status_code == 200
    assert client.get(f"/api/stories/{first['id']}").status_code == 404
    assert client.delete(f"/api/stories/{first['id']}").status_code == 404


def test_validation_errors_are_clean_400s(client):
    cases = [
        {},                                                        # no prompt
        {"prompt": "   "},                                         # blank
        {"prompt": "x" * 201},                                     # over the 200-char test limit
        {"prompt": "hi", "genre": "western"},                      # unknown genre
        {"prompt": "hi", "params": {"temperature": 99}},           # out of range
        {"prompt": "hi", "params": {"max_new_tokens": 61}},        # over the server cap of 60
        {"prompt": "hi", "params": "nope"},                        # wrong type
    ]
    for body in cases:
        r = post(client, "/api/generate", **body)
        assert r.status_code == 400, body
        assert r.get_json()["error"]
    assert client.post("/api/generate", data="not json", content_type="text/plain").status_code == 400
    assert client.get("/api/stories").get_json()["stories"] == []   # nothing was saved


def test_unknown_api_route_returns_json_404(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.get_json()


def test_stream_endpoint_emits_start_tokens_done_and_saves(client):
    r = post(client, "/api/generate/stream", prompt="The old tower", genre="fantasy",
             params={"max_new_tokens": 25, "seed": 4})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    events = parse_sse(r.get_data(as_text=True))
    assert events[0]["type"] == "start" and events[0]["seed"] == 4
    assert events[-1]["type"] == "done"
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert streamed  # something was streamed
    done = events[-1]
    assert done["story"].startswith("The old tower")

    saved = client.get(f"/api/stories/{done['id']}").get_json()
    assert saved["continuation"] == done["continuation"]


def test_stream_endpoint_validation_error_is_plain_400(client):
    assert post(client, "/api/generate/stream", prompt="").status_code == 400


def test_stream_and_normal_endpoint_agree_for_same_seed(client):
    body = dict(prompt="The old tower", params={"max_new_tokens": 30, "seed": 21})
    normal = post(client, "/api/generate", **body).get_json()["results"][0]["continuation"]
    events = parse_sse(post(client, "/api/generate/stream", **body).get_data(as_text=True))
    assert events[-1]["continuation"] == normal
