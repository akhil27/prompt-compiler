import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_HF_BASE_URL = "https://router.huggingface.co/v1"

FREE_MODELS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-3n-E4B-it",
]

DEFAULT_MODEL = FREE_MODELS[0]

last_used_model: str = DEFAULT_MODEL

_FALLBACK_STATUS = {402, 408, 409, 425, 429, 500, 502, 503, 504}
_AUTH_STATUS = {401, 403}

_FALLBACK_SIGNALS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "quota",
    "unavailable",
    "overloaded",
    "not supported",
    "model_not_found",
    "currently loading",
)
_AUTH_SIGNALS = (
    "invalid username",
    "invalid token",
    "unauthorized",
    "forbidden",
)

_client = None


def _resolve_token() -> str:
    """Read the HF token, preferring HF_TOKEN, with HF_API_KEY as legacy fallback."""
    load_dotenv(override=False)
    token = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    return (token or "").strip()


def get_client() -> OpenAI:
    """
    Return a lazily-initialised OpenAI-compatible client pointed at the
    Hugging Face Router. Reads HF_TOKEN on every call so Settings-modal
    changes are picked up without restarting the server.
    """
    global _client

    api_key = _resolve_token()
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN is not set. Open ⚙️ Settings and paste your Hugging "
            "Face token (create one at https://huggingface.co/settings/tokens "
            "with 'Make calls to Inference Providers' enabled)."
        )

    if _client is None or _client.api_key != api_key:
        _client = OpenAI(api_key=api_key, base_url=_HF_BASE_URL)

    return _client


def _status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from an OpenAI/HTTP exception."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_auth_error(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in _AUTH_STATUS:
        return True
    low = str(exc).lower()
    return any(sig in low for sig in _AUTH_SIGNALS)


def _is_fallback_error(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code in _FALLBACK_STATUS:
        return True
    low = str(exc).lower()
    return any(sig in low for sig in _FALLBACK_SIGNALS)


def _call_model(client: OpenAI, model: str, messages, temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60.0,
    )
    return response.choices[0].message.content


def generate(
    prompt: str,
    system_prompt: str = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    Send a chat-completion request to the Hugging Face Router using only
    free-tier models. On rate-limit / quota / availability errors, automatically
    iterate through the remaining models in FREE_MODELS.

    Sets the module-level `last_used_model` to whichever model succeeded.
    Raises a friendly RuntimeError on auth failure or when all models are exhausted.
    """
    global last_used_model

    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    safe_temp = max(0.01, min(1.2, temperature))

    candidates = [model] + [m for m in FREE_MODELS if m != model]

    last_error = None
    tried = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            result = _call_model(client, candidate, messages, safe_temp, max_tokens)
            last_used_model = candidate
            return result
        except Exception as e:
            last_error = e

            if _is_auth_error(e):
                raise RuntimeError(
                    "Hugging Face token rejected (401/403). Your HF_TOKEN is "
                    "invalid or lacks 'Make calls to Inference Providers' "
                    "permission. Create a new fine-grained token at "
                    "https://huggingface.co/settings/tokens and paste it in "
                    "⚙️ Settings."
                ) from e

            if _is_fallback_error(e):
                continue

            raise RuntimeError(f"Hugging Face API error: {e}") from e

    tried_list = "\n  - ".join(tried)
    raise RuntimeError(
        "All free-tier Hugging Face models are currently rate-limited or "
        f"unavailable. Tried:\n  - {tried_list}\n\nLast error: {last_error}"
    )
