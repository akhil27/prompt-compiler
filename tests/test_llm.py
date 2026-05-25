import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import llm
from llm import (
    _resolve_token,
    _status_code,
    _is_auth_error,
    _is_fallback_error,
    FREE_MODELS,
    DEFAULT_MODEL,
    generate,
    get_client,
)


def make_openai_error(status_code=None, message="error"):
    exc = Exception(message)
    if status_code is not None:
        exc.status_code = status_code
    return exc


def test_free_models_is_nonempty_list():
    assert isinstance(FREE_MODELS, list)
    assert len(FREE_MODELS) > 0


def test_default_model_is_first_in_free_models():
    assert DEFAULT_MODEL == FREE_MODELS[0]


def test_resolve_token_reads_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_testtoken123")
    token = _resolve_token()
    assert token == "hf_testtoken123"


def test_resolve_token_falls_back_to_hf_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_API_KEY", "hf_legacykey")
    with patch("llm.load_dotenv"):
        token = _resolve_token()
    assert token == "hf_legacykey"


def test_resolve_token_prefers_hf_token_over_hf_api_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_primary")
    monkeypatch.setenv("HF_API_KEY", "hf_legacy")
    token = _resolve_token()
    assert token == "hf_primary"


def test_resolve_token_strips_whitespace(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_padded  ")
    token = _resolve_token()
    assert token == "hf_padded"


def test_resolve_token_returns_empty_when_not_set(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    with patch("llm.load_dotenv"):
        token = _resolve_token()
    assert token == ""


def test_status_code_reads_status_code_attr():
    exc = Exception("bad")
    exc.status_code = 429
    assert _status_code(exc) == 429


def test_status_code_reads_response_status_code():
    exc = Exception("bad")
    response = MagicMock()
    response.status_code = 503
    exc.response = response
    assert _status_code(exc) == 503


def test_status_code_returns_none_when_absent():
    exc = Exception("no status")
    assert _status_code(exc) is None


def test_is_auth_error_true_for_401():
    exc = make_openai_error(status_code=401)
    assert _is_auth_error(exc)


def test_is_auth_error_true_for_403():
    exc = make_openai_error(status_code=403)
    assert _is_auth_error(exc)


def test_is_auth_error_true_for_unauthorized_message():
    exc = Exception("Unauthorized access to model")
    assert _is_auth_error(exc)


def test_is_auth_error_true_for_invalid_token_message():
    exc = Exception("Invalid token provided")
    assert _is_auth_error(exc)


def test_is_auth_error_false_for_429():
    exc = make_openai_error(status_code=429)
    assert not _is_auth_error(exc)


def test_is_auth_error_false_for_generic_error():
    exc = Exception("Something went wrong")
    assert not _is_auth_error(exc)


def test_is_fallback_error_true_for_429():
    exc = make_openai_error(status_code=429)
    assert _is_fallback_error(exc)


def test_is_fallback_error_true_for_503():
    exc = make_openai_error(status_code=503)
    assert _is_fallback_error(exc)


def test_is_fallback_error_true_for_502():
    exc = make_openai_error(status_code=502)
    assert _is_fallback_error(exc)


def test_is_fallback_error_true_for_rate_limit_message():
    exc = Exception("rate limit exceeded")
    assert _is_fallback_error(exc)


def test_is_fallback_error_true_for_quota_message():
    exc = Exception("quota exhausted for this model")
    assert _is_fallback_error(exc)


def test_is_fallback_error_true_for_overloaded_message():
    exc = Exception("model is currently overloaded")
    assert _is_fallback_error(exc)


def test_is_fallback_error_false_for_401():
    exc = make_openai_error(status_code=401)
    assert not _is_fallback_error(exc)


def test_is_fallback_error_false_for_generic_message():
    exc = Exception("unexpected value in response")
    assert not _is_fallback_error(exc)


def test_get_client_raises_when_no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_API_KEY", raising=False)
    llm._client = None
    with patch("llm.load_dotenv"):
        with pytest.raises(RuntimeError, match="HF_TOKEN"):
            get_client()


def test_get_client_returns_client_with_valid_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_validtoken")
    llm._client = None
    with patch("llm.OpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_instance.api_key = "hf_validtoken"
        mock_openai.return_value = mock_instance
        client = get_client()
    assert client is not None


def test_generate_returns_string_on_success(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    llm._client = None
    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "## GOAL\nBuild API"
    mock_client.chat.completions.create.return_value = mock_response
    with patch("llm.get_client", return_value=mock_client):
        result = generate("make an API")
    assert result == "## GOAL\nBuild API"


def test_generate_falls_back_on_rate_limit(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            exc = Exception("rate limit exceeded")
            exc.status_code = 429
            raise exc
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "## GOAL\nFallback output"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        result = generate("make api", model=FREE_MODELS[0])
    assert "Fallback output" in result
    assert call_count["n"] == 2


def test_generate_raises_auth_error_immediately(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_invalid")
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        exc = Exception("Unauthorized")
        exc.status_code = 401
        raise exc

    mock_client = MagicMock()
    mock_client.api_key = "hf_invalid"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="rejected"):
            generate("make api")
    assert call_count["n"] == 1


def test_generate_raises_when_all_models_exhausted(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def fake_create(**kwargs):
        exc = Exception("rate limit exceeded")
        exc.status_code = 429
        raise exc

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="rate-limited or unavailable"):
            generate("make api")


def test_generate_raises_unknown_error_without_fallback(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        raise Exception("unexpected json decode error")

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="Hugging Face API error"):
            generate("make api")
    assert call_count["n"] == 1


def test_generate_tries_requested_model_first(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    tried_models = []

    def fake_create(**kwargs):
        tried_models.append(kwargs["model"])
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("test", model="google/gemma-3n-E4B-it")

    assert tried_models[0] == "google/gemma-3n-E4B-it"


def test_generate_includes_system_prompt_when_provided(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    captured_messages = []

    def fake_create(**kwargs):
        captured_messages.extend(kwargs["messages"])
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("user input", system_prompt="You are an expert.")

    roles = [m["role"] for m in captured_messages]
    assert "system" in roles


def test_generate_no_system_prompt_only_user_message(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    captured_messages = []

    def fake_create(**kwargs):
        captured_messages.extend(kwargs["messages"])
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("user input only")

    roles = [m["role"] for m in captured_messages]
    assert roles == ["user"]


def test_generate_temperature_clipped_below_minimum(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    captured_temp = []

    def fake_create(**kwargs):
        captured_temp.append(kwargs["temperature"])
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("test", temperature=0.0)

    assert captured_temp[0] >= 0.01


def test_generate_temperature_clipped_above_maximum(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    captured_temp = []

    def fake_create(**kwargs):
        captured_temp.append(kwargs["temperature"])
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "ok"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("test", temperature=9.9)

    assert captured_temp[0] <= 1.2


def test_last_used_model_set_on_success(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    def fake_create(**kwargs):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "compiled output"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("test", model=FREE_MODELS[0])

    assert llm.last_used_model == FREE_MODELS[0]


def test_last_used_model_set_to_fallback_when_primary_fails(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            exc = Exception("rate limit exceeded")
            exc.status_code = 429
            raise exc
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "fallback output"
        return mock_response

    mock_client = MagicMock()
    mock_client.api_key = "hf_test"
    mock_client.chat.completions.create.side_effect = fake_create

    with patch("llm.get_client", return_value=mock_client):
        generate("test", model=FREE_MODELS[0])

    assert llm.last_used_model == FREE_MODELS[1]


def test_free_models_has_three_entries():
    assert len(FREE_MODELS) == 3


def test_last_used_model_is_string():
    assert isinstance(llm.last_used_model, str)
    assert len(llm.last_used_model) > 0

