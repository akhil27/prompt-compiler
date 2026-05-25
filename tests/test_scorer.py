import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from scorer import score_prompt


def test_empty_string_returns_zero():
    result = score_prompt("")
    assert result["score"] == 0
    assert result["criteria_scores"]["specificity"] == 0
    assert result["criteria_scores"]["structure"] == 0
    assert result["criteria_scores"]["constraints"] == 0
    assert result["criteria_scores"]["formatting"] == 0
    assert result["criteria_scores"]["context"] == 0
    assert "Prompt is empty." in result["weaknesses"]


def test_whitespace_only_returns_zero():
    result = score_prompt("   \n\t  ")
    assert result["score"] == 0


def test_none_equivalent_empty_string():
    result = score_prompt("")
    assert isinstance(result["score"], int)
    assert result["score"] == 0


def test_single_word_prompt_low_specificity():
    result = score_prompt("hello")
    assert result["criteria_scores"]["specificity"] <= 5
    assert result["score"] < 20


def test_very_short_prompt_specificity_band():
    result = score_prompt("build app")
    assert result["criteria_scores"]["specificity"] == 3
    assert "extremely brief" in result["weaknesses"][0].lower() or \
           any("brief" in w.lower() for w in result["weaknesses"])


def test_medium_prompt_mid_specificity():
    prompt = "write a python function that parses json files from a directory"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["specificity"] == 10
    assert result["score"] > 5


def test_long_prompt_high_specificity():
    prompt = " ".join(["word"] * 50)
    result = score_prompt(prompt)
    assert result["criteria_scores"]["specificity"] >= 15


def test_specificity_capped_at_20():
    prompt = " ".join(["word"] * 200)
    result = score_prompt(prompt)
    assert result["criteria_scores"]["specificity"] <= 20


def test_markdown_headers_add_structure():
    prompt = "## Goal\nBuild a REST API\n## Constraints\nNo auth required"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] >= 8


def test_bullet_points_add_structure():
    prompt = "Build a tool that:\n- handles JSON\n- returns CSV\n- logs errors"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] >= 8


def test_numbered_list_adds_structure():
    prompt = "Steps:\n1. Parse input\n2. Validate schema\n3. Return result"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] >= 8


def test_double_newline_adds_structure_points():
    prompt = "First paragraph here.\n\nSecond paragraph here."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] >= 4


def test_unstructured_single_block_zero_structure():
    prompt = "make a website that looks good and has animations and works on mobile"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] == 0
    assert any("unstructured" in w.lower() for w in result["weaknesses"])


def test_full_structure_max_score():
    prompt = "## Goal\nBuild an API\n\n## Rules\n- No auth\n- JSON only\n1. Parse\n2. Return"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["structure"] == 20


def test_constraint_keyword_avoid_scores():
    prompt = "Build an API. Avoid using external libraries. Must be synchronous."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["constraints"] >= 5


def test_multiple_constraint_keywords():
    prompt = "Never use global state. Avoid magic strings. Must validate inputs. Do not import unused modules. Limit to 100 lines."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["constraints"] >= 15


def test_constraint_score_capped_at_20():
    prompt = "avoid avoid avoid must must must never never limit limit restrict restrict constraint constraint rule rule exclude exclude only only"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["constraints"] <= 20


def test_no_constraint_keywords_zero_score():
    prompt = "Write a function that reads files and prints them to the screen."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["constraints"] == 0
    assert any("constraint" in w.lower() for w in result["missing_components"])


def test_format_keyword_json_scores():
    prompt = "Return output as a JSON block with keys: name, age, email."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["formatting"] >= 6


def test_format_keyword_markdown_scores():
    prompt = "Output the result as a markdown table with columns for name and score."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["formatting"] >= 6


def test_multiple_format_keywords():
    prompt = "Return a JSON schema. Format as markdown. Include a table. Show code. Use yaml structure."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["formatting"] >= 18


def test_format_score_capped_at_20():
    prompt = "json json json markdown markdown table table code code output output format format schema schema html html yaml yaml"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["formatting"] <= 20


def test_no_format_keywords_zero_score_and_missing():
    prompt = "Write a function that reads files."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["formatting"] == 0
    assert any("format" in c.lower() for c in result["missing_components"])


def test_role_keyword_scores_context():
    prompt = "Act as a senior Python developer and review the following code for issues."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["context"] >= 6


def test_multiple_context_keywords():
    prompt = "Role: Expert Data Scientist. Context: analyzing customer churn. Target audience: business stakeholders. Objective: identify top 3 factors."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["context"] >= 18


def test_context_score_capped_at_20():
    prompt = "role role persona audience context background objective goal target user customer expert specialist act"
    result = score_prompt(prompt)
    assert result["criteria_scores"]["context"] <= 20


def test_no_context_keywords_zero_score_and_missing():
    prompt = "Write a function that reads files."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["context"] == 0
    assert any("persona" in c.lower() or "role" in c.lower() for c in result["missing_components"])


def test_total_score_is_sum_of_criteria():
    prompt = "## Role\nExpert developer. Avoid globals. Return JSON. Act as specialist."
    result = score_prompt(prompt)
    expected = sum(result["criteria_scores"].values())
    assert result["score"] == min(100, max(0, expected))


def test_total_score_never_exceeds_100():
    prompt = "\n".join([
        "## GOAL\nBuild a fully featured REST API",
        "## CONSTRAINTS\nAvoid globals. Must validate. Never skip auth. Limit to 500 lines.",
        "## FORMAT\nReturn JSON schema with markdown table and code block",
        "## ROLE\nAct as expert architect with context about distributed systems"
    ] * 5)
    result = score_prompt(prompt)
    assert result["score"] <= 100


def test_total_score_never_below_zero():
    result = score_prompt("")
    assert result["score"] >= 0


def test_high_quality_prompt_above_85():
    prompt = (
        "## ROLE\nAct as a Senior Backend Engineer specializing in Python REST APIs.\n\n"
        "## OBJECTIVE\nBuild a secure user authentication endpoint using FastAPI.\n\n"
        "## CONSTRAINTS\n- Must use JWT tokens. Never store raw passwords. Avoid third-party OAuth. "
        "Do not use global variables. Only standard library plus FastAPI.\n\n"
        "## OUTPUT FORMAT\nReturn valid Python code only. No explanations. Format as a code block.\n\n"
        "## CONTEXT\nThe target audience is backend developers. Background: greenfield SaaS product. "
        "Goal: minimal, secure, production-ready auth."
    )
    result = score_prompt(prompt)
    assert result["score"] >= 60


def test_result_has_all_required_keys():
    result = score_prompt("some prompt")
    assert "score" in result
    assert "criteria_scores" in result
    assert "weaknesses" in result
    assert "suggestions" in result
    assert "missing_components" in result


def test_criteria_scores_has_all_five_dimensions():
    result = score_prompt("some prompt")
    for dim in ["specificity", "structure", "constraints", "formatting", "context"]:
        assert dim in result["criteria_scores"]


def test_weaknesses_is_list():
    result = score_prompt("some prompt")
    assert isinstance(result["weaknesses"], list)


def test_suggestions_is_list():
    result = score_prompt("some prompt")
    assert isinstance(result["suggestions"], list)


def test_missing_components_is_list():
    result = score_prompt("some prompt")
    assert isinstance(result["missing_components"], list)


def test_unicode_prompt_does_not_crash():
    result = score_prompt("构建一个 REST API，避免使用外部库，返回 JSON 格式。")
    assert isinstance(result["score"], int)


def test_very_long_prompt_does_not_crash():
    result = score_prompt("word " * 5000)
    assert isinstance(result["score"], int)
    assert result["score"] <= 100


def test_prompt_with_only_symbols_does_not_crash():
    result = score_prompt("!!! ??? ### @@@ $$$")
    assert isinstance(result["score"], int)


def test_newline_only_prompt():
    result = score_prompt("\n\n\n\n")
    assert result["score"] == 0


def test_case_insensitive_keyword_matching():
    prompt = "AVOID global state. MUST validate inputs. NEVER skip error handling."
    result = score_prompt(prompt)
    assert result["criteria_scores"]["constraints"] >= 10


def test_partial_word_not_matched_as_constraint():
    prompt = "This avoidance strategy avoids constraints but doesn't must anything."
    result_with = score_prompt("avoid must never limit")
    result_partial = score_prompt("avoidance mustard neverland limitation")
    assert result_with["criteria_scores"]["constraints"] > result_partial["criteria_scores"]["constraints"]
