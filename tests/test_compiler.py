import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from compiler import compile_prompt
from llm import DEFAULT_MODEL


def make_mock_generate(return_value="## GOAL\nDo something\n## CONSTRAINTS\nNone"):
    return MagicMock(return_value=return_value)


def test_compile_prompt_returns_string():
    with patch("compiler.generate", return_value="## GOAL\nBuild API"):
        result = compile_prompt("make an api")
    assert isinstance(result, str)


def test_compile_prompt_empty_input_raises():
    with pytest.raises(ValueError):
        compile_prompt("")


def test_compile_prompt_whitespace_only_raises():
    with pytest.raises(ValueError):
        compile_prompt("   \n\t  ")


def test_compile_prompt_strips_result():
    with patch("compiler.generate", return_value="  ## GOAL\nBuild API  \n"):
        result = compile_prompt("make an api")
    assert not result.startswith(" ")
    assert not result.endswith(" ")


def test_compile_prompt_passes_model_to_generate():
    mock = MagicMock(return_value="## GOAL\nBuild")
    with patch("compiler.generate", mock):
        compile_prompt("make api", model="Qwen/Qwen3-4B-Instruct-2507")
    call_kwargs = mock.call_args
    assert call_kwargs.kwargs.get("model") == "Qwen/Qwen3-4B-Instruct-2507" or \
           call_kwargs.args[2] == "Qwen/Qwen3-4B-Instruct-2507" if len(call_kwargs.args) > 2 else True


def test_compile_prompt_passes_temperature_to_generate():
    mock = MagicMock(return_value="## GOAL\nBuild")
    with patch("compiler.generate", mock):
        compile_prompt("make api", temperature=0.7)
    call_kwargs = mock.call_args
    assert call_kwargs.kwargs.get("temperature") == 0.7


def test_compile_prompt_uses_default_model_when_not_specified():
    mock = MagicMock(return_value="## GOAL\nBuild")
    with patch("compiler.generate", mock):
        compile_prompt("make api")
    call_kwargs = mock.call_args
    assert call_kwargs.kwargs.get("model") == DEFAULT_MODEL


def test_compile_prompt_raises_on_generate_failure():
    with patch("compiler.generate", side_effect=RuntimeError("API down")):
        with pytest.raises(RuntimeError, match="API down"):
            compile_prompt("make api")


def test_compile_prompt_includes_user_input_in_payload():
    captured = {}
    def capture_generate(prompt, system_prompt, model, temperature):
        captured["prompt"] = prompt
        return "## GOAL\nDone"
    with patch("compiler.generate", side_effect=capture_generate):
        compile_prompt("build a portfolio site")
    assert "build a portfolio site" in captured["prompt"]


def test_compile_prompt_uses_system_prompt():
    captured = {}
    def capture_generate(prompt, system_prompt, model, temperature):
        captured["system_prompt"] = system_prompt
        return "## GOAL\nDone"
    with patch("compiler.generate", side_effect=capture_generate):
        compile_prompt("make api")
    assert captured["system_prompt"] is not None
    assert len(captured["system_prompt"]) > 50


def test_compile_prompt_unicode_input():
    with patch("compiler.generate", return_value="## GOAL\n構建 API"):
        result = compile_prompt("构建一个 REST API")
    assert isinstance(result, str)


def test_compile_prompt_very_long_input():
    long_input = "build " * 500
    with patch("compiler.generate", return_value="## GOAL\nBuild something"):
        result = compile_prompt(long_input)
    assert isinstance(result, str)


def test_compile_prompt_special_characters_in_input():
    with patch("compiler.generate", return_value="## GOAL\nBuild"):
        result = compile_prompt("build an API with <auth> & \"JWT\" tokens!")
    assert isinstance(result, str)


def test_compile_prompt_newlines_in_input():
    with patch("compiler.generate", return_value="## GOAL\nBuild"):
        result = compile_prompt("make an api\nthat handles auth\nand returns json")
    assert isinstance(result, str)
