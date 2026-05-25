import re
from typing import Dict, List, Any

def score_prompt(prompt: str) -> Dict[str, Any]:
    """
    Evaluates a prompt using local heuristics.
    Returns a dictionary detailing:
    - score: int (0 to 100)
    - criteria_scores: dict (specificity, structure, constraints, formatting, context)
    - weaknesses: list of strings
    - suggestions: list of strings
    - missing_components: list of strings
    """
    if not prompt or not prompt.strip():
        return {
            "score": 0,
            "criteria_scores": {
                "specificity": 0,
                "structure": 0,
                "constraints": 0,
                "formatting": 0,
                "context": 0
            },
            "weaknesses": ["Prompt is empty."],
            "suggestions": ["Please enter a prompt to begin scoring."],
            "missing_components": ["Everything is missing."]
        }

    text = prompt.strip()
    words = text.split()
    word_count = len(words)

    # Criteria scores mapping (Max 20 each, total 100)
    criteria = {
        "specificity": 0,
        "structure": 0,
        "constraints": 0,
        "formatting": 0,
        "context": 0
    }

    weaknesses = []
    suggestions = []
    missing_components = []

    # 1. SPECIFICITY (Based on length and vocabulary size)
    if word_count < 5:
        criteria["specificity"] = 3
        weaknesses.append("The prompt is extremely brief and lacks specificity.")
        suggestions.append("Expand your prompt. Describe the specific project, topic, or task in more detail (aim for at least 15-20 words).")
        missing_components.append("Elaborative task context")
    elif word_count < 15:
        criteria["specificity"] = 10
        weaknesses.append("The prompt is short and might yield generic AI outputs.")
        suggestions.append("Provide details about the target audience, specific features, or tone requirements.")
    else:
        # Scale specificity between 12 and 20 based on word count
        criteria["specificity"] = min(20, 10 + int(word_count / 10))

    # 2. STRUCTURE (Headers, bullet lists, spacing)
    has_headers = bool(re.search(r'^#+\s+\w+', text, re.MULTILINE))
    has_bullets = bool(re.search(r'^[\*\-\+]\s+', text, re.MULTILINE))
    has_numbered = bool(re.search(r'^\d+\.\s+', text, re.MULTILINE))
    has_newlines = "\n\n" in text or "\r\n\r\n" in text

    if has_headers:
        criteria["structure"] += 8
    if has_bullets or has_numbered:
        criteria["structure"] += 8
    if has_newlines:
        criteria["structure"] += 4

    if criteria["structure"] == 0:
        weaknesses.append("Prompt is a single block of unstructured text.")
        suggestions.append("Organize your prompt using Markdown headers (## Role, ## Constraints) or clear paragraphs.")
        missing_components.append("Clear section layout")
    elif criteria["structure"] < 12:
        weaknesses.append("The structure is minimal.")
        suggestions.append("Add bullet points or numbered lists to make requirements easier for the AI to digest.")

    # 3. CONSTRAINTS (Negative instructions, boundaries)
    constraint_keywords = [
        "avoid", "don't", "dont", "must", "never", "limit", "bound", "restrict",
        "constraint", "negative", "rule", "should not", "do not", "exclude", "only"
    ]
    constraint_matches = sum(1 for kw in constraint_keywords if re.search(r'\b' + kw + r'\b', text.lower()))
    
    criteria["constraints"] = min(20, constraint_matches * 5)
    if criteria["constraints"] == 0:
        weaknesses.append("No clear negative constraints or boundaries are set.")
        suggestions.append("Add constraints to restrict the AI. What should it avoid? (e.g. 'no boilerplate code', 'avoid buzzwords').")
        missing_components.append("Negative constraints (what to avoid)")

    # 4. FORMATTING (Desired output format)
    format_keywords = [
        "format", "json", "markdown", "csv", "xml", "output", "template", "table",
        "yaml", "html", "syntax", "schema", "style", "structure", "list", "code"
    ]
    format_matches = sum(1 for kw in format_keywords if re.search(r'\b' + kw + r'\b', text.lower()))
    
    criteria["formatting"] = min(20, format_matches * 6)
    if criteria["formatting"] == 0:
        weaknesses.append("No output formatting instructions are specified.")
        suggestions.append("Define the desired output style (e.g. 'return only a JSON block', 'format as a markdown table').")
        missing_components.append("Output formatting specification")

    # 5. CONTEXT & PERSONA (Who is the model, background info)
    context_keywords = [
        "role", "persona", "audience", "context", "background", "objective", "goal",
        "target", "situation", "user", "customer", "expert", "specialist", "act as"
    ]
    context_matches = sum(1 for kw in context_keywords if re.search(r'\b' + kw + r'\b', text.lower()))

    criteria["context"] = min(20, context_matches * 6)
    if criteria["context"] == 0:
        weaknesses.append("The prompt lacks background context or an assigned AI persona.")
        suggestions.append("Assign a role to the AI (e.g. 'Act as a Senior Frontend Developer') to set the right level of expertise.")
        missing_components.append("Role / Persona definition")

    # Compute final combined score (max 100)
    total_score = sum(criteria.values())
    total_score = max(0, min(100, total_score))

    # Adjust recommendations for excellent prompts
    if total_score >= 85:
        if not weaknesses:
            weaknesses.append("None. The prompt is highly optimized!")
        if not suggestions:
            suggestions.append("Your prompt is structural and precise. Ready to run.")
        if not missing_components:
            missing_components.append("None.")

    return {
        "score": total_score,
        "criteria_scores": criteria,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "missing_components": missing_components
    }
