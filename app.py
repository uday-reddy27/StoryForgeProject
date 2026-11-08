"""Run the web app:  python app.py   (then open http://127.0.0.1:5000)"""

import logging
import os

from storyforge.web import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

if __name__ == "__main__":
    # threaded=True lets the page poll /api/health while a story is being generated.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        threaded=True,
        debug=False,
    )
