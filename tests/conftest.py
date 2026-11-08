import pytest

from storyforge.config import Settings
from storyforge.generator import StoryGenerator
from storyforge.storage import StoryStore
from storyforge.web import create_app
from tiny_model import build_tiny_model


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return build_tiny_model(tmp_path_factory.mktemp("tiny-gpt2"))


@pytest.fixture(scope="session")
def generator(tiny_model_dir):
    gen = StoryGenerator(str(tiny_model_dir), device="cpu")
    gen.load()
    return gen


@pytest.fixture()
def client(generator, tmp_path):
    settings = Settings(model_name="tiny", db_path=tmp_path / "test.db",
                        max_new_tokens_cap=60, max_prompt_chars=200)
    app = create_app(settings, generator=generator, store=StoryStore(settings.db_path))
    app.config["TESTING"] = True
    return app.test_client()
