import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils import parse_sections, extract_final_prompt, generate_diff_markdown, save_output
import json
import tempfile


def test_parse_sections_basic():
    text = "## GOAL\nBuild an API\n\n## CONSTRAINTS\nNo auth"
    result = parse_sections(text)
    assert "goal" in result
    assert "constraints" in result
    assert "Build an API" in result["goal"]
    assert "No auth" in result["constraints"]


def test_parse_sections_empty_string():
    result = parse_sections("")
    assert isinstance(result, dict)


def test_parse_sections_no_headers():
    text = "Just a plain prompt with no markdown headers."
    result = parse_sections(text)
    assert "overview" in result
    assert "Just a plain prompt" in result["overview"]


def test_parse_sections_header_names_lowercased():
    text = "## FINAL OPTIMIZED PROMPT\nSome final text"
    result = parse_sections(text)
    assert "final_optimized_prompt" in result


def test_parse_sections_special_chars_in_header_normalized():
    text = "## OUTPUT FORMAT\nReturn JSON"
    result = parse_sections(text)
    assert "output_format" in result


def test_parse_sections_multiple_sections():
    text = "## GOAL\nGoal here\n## TONE\nProfessional\n## CONSTRAINTS\nNone"
    result = parse_sections(text)
    assert len(result) >= 3
    assert "goal" in result
    assert "tone" in result
    assert "constraints" in result


def test_parse_sections_content_stripped():
    text = "## GOAL\n  \nBuild something\n  "
    result = parse_sections(text)
    assert result["goal"] == "Build something"


def test_parse_sections_empty_section_body():
    text = "## GOAL\n## CONSTRAINTS\nSome constraint"
    result = parse_sections(text)
    assert "goal" in result
    assert result["goal"] == ""


def test_extract_final_prompt_present():
    text = "## GOAL\nBuild API\n## FINAL OPTIMIZED PROMPT\nAct as senior dev. Build it."
    result = extract_final_prompt(text)
    assert result == "Act as senior dev. Build it."
    assert "## GOAL" not in result


def test_extract_final_prompt_missing_returns_full_text():
    text = "## GOAL\nBuild API\n## CONSTRAINTS\nNo auth"
    result = extract_final_prompt(text)
    assert result == text


def test_extract_final_prompt_case_insensitive():
    text = "## final optimized prompt\nSome final content here."
    result = extract_final_prompt(text)
    assert result == "Some final content here."


def test_extract_final_prompt_empty_section():
    text = "## FINAL OPTIMIZED PROMPT\n"
    result = extract_final_prompt(text)
    assert result == ""


def test_extract_final_prompt_multiline():
    text = "Preamble\n## FINAL OPTIMIZED PROMPT\nLine one\nLine two\nLine three"
    result = extract_final_prompt(text)
    assert "Line one" in result
    assert "Line two" in result
    assert "Line three" in result


def test_generate_diff_markdown_returns_diff_block():
    original = "make a website"
    compiled = "## GOAL\nBuild a responsive website"
    result = generate_diff_markdown(original, compiled)
    assert result.startswith("```diff")
    assert result.strip().endswith("```")


def test_generate_diff_markdown_shows_additions():
    original = "make a website"
    compiled = "make a beautiful responsive website with animations"
    result = generate_diff_markdown(original, compiled)
    assert "+" in result


def test_generate_diff_markdown_shows_deletions():
    original = "make a big slow website"
    compiled = "make a fast website"
    result = generate_diff_markdown(original, compiled)
    assert "-" in result


def test_generate_diff_markdown_identical_inputs():
    text = "same text on both sides"
    result = generate_diff_markdown(text, text)
    assert "```diff" in result


def test_generate_diff_markdown_empty_original():
    result = generate_diff_markdown("", "some compiled output")
    assert "```diff" in result


def test_generate_diff_markdown_empty_compiled():
    result = generate_diff_markdown("some original text", "")
    assert "```diff" in result


def test_generate_diff_markdown_both_empty():
    result = generate_diff_markdown("", "")
    assert "```diff" in result


def test_save_output_json_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("test prompt", "## GOAL\nBuild API", "json")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["original_prompt"] == "test prompt"
    assert data["compiled_prompt"] == "## GOAL\nBuild API"
    assert "timestamp" in data
    assert data["version"] == "2.0"


def test_save_output_md_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("test prompt", "## GOAL\nBuild API", "md")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Compiled Prompt Output" in content
    assert "test prompt" in content


def test_save_output_txt_creates_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("test prompt", "## GOAL\nBuild API", "txt")
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "ORIGINAL PROMPT" in content
    assert "COMPILED PROMPT" in content


def test_save_output_filename_slug_from_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("make landing page", "compiled", "json")
    assert "make_landing_page" in os.path.basename(path)


def test_save_output_empty_original_uses_fallback_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("", "some compiled output", "json")
    assert os.path.exists(path)
    assert "compiled" in os.path.basename(path)


def test_save_output_special_chars_in_prompt_slugified(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("hello! @world#2024", "compiled output", "json")
    basename = os.path.basename(path)
    assert " " not in basename
    assert "!" not in basename
    assert "@" not in basename


def test_save_output_json_contains_parsed_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    compiled = "## GOAL\nBuild REST API\n## CONSTRAINTS\nNo auth"
    path = save_output("raw", compiled, "json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "parsed_sections" in data
    assert "goal" in data["parsed_sections"]


def test_save_output_unicode_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_output("构建 API", "## GOAL\nBuild something", "json")
    assert os.path.exists(path)
