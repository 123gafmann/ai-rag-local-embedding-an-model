import logging
from pathlib import Path

from .config import LOG_LEVEL

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(_formatter)

app_file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
app_file_handler.setLevel(logging.INFO)
app_file_handler.setFormatter(_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(app_file_handler)

# These libraries log every HTTP call at INFO, which drowns out our own
# logs without adding anything useful day-to-day.
for noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Full LLM prompts/responses are too verbose for the console but useful to
# have on disk when debugging a bad answer, so they get their own file and
# don't propagate to the console handler above.
llm_io_handler = logging.FileHandler(LOG_DIR / "llm_calls.log", encoding="utf-8")
llm_io_handler.setLevel(logging.INFO)
llm_io_handler.setFormatter(_formatter)

llm_io_logger = logging.getLogger("llm_io")
llm_io_logger.setLevel(logging.INFO)
llm_io_logger.addHandler(llm_io_handler)
llm_io_logger.propagate = False
