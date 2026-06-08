"""Central LLM API client for the customer-support agent prototype.

Wraps google-genai to call Gemini 2.0 Flash. All modules that need to make LLM
calls (NLU, response generation, escalation reasoning, rubric scoring) must
import through this module so that setup, caching, and error handling are shared.

The module reads GOOGLE_API_KEY from the environment at import time (via
python-dotenv so a .env file at the repo root is automatically picked up).
If the key is absent, LLM_AVAILABLE is set to False and every call to
complete() raises LLMClientError immediately.

Responses are cached to .cache/llm_responses.json by default to avoid
redundant API calls during development. Set DISABLE_LLM_CACHE=1 to bypass.
"""

import hashlib
import json
import os
import pathlib
import time

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

_REPO_ROOT = pathlib.Path(__file__).parent.parent
CACHE_PATH = _REPO_ROOT / ".cache" / "llm_responses.json"

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY:
    _client = genai.Client(api_key=GOOGLE_API_KEY)
    LLM_AVAILABLE = True
else:
    _client = None
    LLM_AVAILABLE = False


class LLMClientError(RuntimeError):
    """Raised when an LLM API call fails for any reason."""


def _load_cache() -> dict[str, str]:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(system: str, user: str, json_mode: bool) -> str:
    raw = f"{system}\x00{user}\x00{json_mode}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def complete(
    system: str,
    user: str,
    json_mode: bool = False,
    model: str = "gemini-2.5-flash",
) -> str:
    """Call the Gemini model and return the response text.

    Raises LLMClientError on any failure (missing key, network error, etc.).
    """
    if not LLM_AVAILABLE or _client is None:
        raise LLMClientError("GOOGLE_API_KEY is not configured")

    caching_enabled = os.environ.get("DISABLE_LLM_CACHE", "").strip() != "1"
    key = _cache_key(system, user, json_mode)

    if caching_enabled:
        cache = _load_cache()
        if key in cache:
            return cache[key]

    config = types.GenerateContentConfig(system_instruction=system)
    if json_mode:
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
        )

    _MAX_RETRIES = 3
    _RETRY_DELAY = 5  # seconds between retries on 503
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = _client.models.generate_content(
                model=model,
                contents=user,
                config=config,
            )
            text: str = response.text
            break
        except Exception as exc:
            last_exc = exc
            if "503" in str(exc) and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
                continue
            raise LLMClientError(f"Gemini API call failed: {exc}") from exc
    else:
        raise LLMClientError(f"Gemini API call failed after {_MAX_RETRIES} attempts: {last_exc}")

    if caching_enabled:
        cache[key] = text
        _save_cache(cache)

    return text
