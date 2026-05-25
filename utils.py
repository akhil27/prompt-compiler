import os
import json
import time
import re
import difflib
from typing import Dict, Any

def ensure_dir(path: str):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def generate_diff_markdown(original: str, compiled: str) -> str:
    """
    Generates a standard diff format string inside a markdown code block.
    This creates the visual diff showing insertions and deletions.
    """
    orig_lines = original.strip().splitlines()
    comp_lines = compiled.strip().splitlines()
    
    # We prefix a header block for visual satisfaction
    diff = difflib.ndiff(orig_lines, comp_lines)
    
    cleaned_diff = []
    for line in diff:
        # Match standard diff notation:
        # '- ' for deletions
        # '+ ' for additions
        # '? ' for guide lines (skip these to avoid clutter)
        if line.startswith('? '):
            continue
        cleaned_diff.append(line)
        
    diff_str = "\n".join(cleaned_diff)
    return f"```diff\n{diff_str}\n```"

def save_output(original: str, compiled: str, format_type: str) -> str:
    """
    Saves compiled prompt results to the 'outputs/' directory.
    
    Parameters:
        original (str): Raw input prompt.
        compiled (str): Structured compiled prompt.
        format_type (str): Output extension ('md', 'txt', or 'json').
        
    Returns:
        str: Relative path to the saved file.
    """
    ensure_dir("outputs")
    
    # Generate clean filename based on original input and timestamp
    slug = re.sub(r'[^a-zA-Z0-9]', '_', original[:20].lower().strip())
    if not slug:
        slug = "compiled"
    timestamp = int(time.time())
    filename = f"prompt_{slug}_{timestamp}.{format_type}"
    filepath = os.path.join("outputs", filename)
    
    if format_type == "json":
        payload = {
            "original_prompt": original,
            "compiled_prompt": compiled,
            "parsed_sections": parse_sections(compiled),
            "timestamp": timestamp,
            "version": "2.0"
        }
        content = json.dumps(payload, indent=2)
    elif format_type == "txt":
        content = (
            f"=== ORIGINAL PROMPT ===\n{original}\n\n"
            f"=== COMPILED PROMPT ===\n\n{compiled}\n"
        )
    else:  # Markdown (.md)
        content = (
            f"# Compiled Prompt Output\n\n"
            f"- **Original Vague Prompt:** `{original}`\n"
            f"- **Compiled Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n\n"
            f"---\n\n"
            f"{compiled}\n"
        )
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    return filepath

def get_history_list() -> list:
    """
    Scans the outputs directory and returns a sorted list of previously saved files.
    """
    if not os.path.exists("outputs"):
        return []
    
    files = []
    for f in os.listdir("outputs"):
        if f.startswith("prompt_") and f.endswith(".json"):
            filepath = os.path.join("outputs", f)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    data["filename"] = f
                    files.append(data)
            except Exception:
                pass
                
    # Sort by timestamp descending (newest first)
    files.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return files

def parse_sections(compiled_text: str) -> Dict[str, str]:
    """
    Parses the compiled prompt text structured by Markdown headers (e.g. ## GOAL)
    into a dictionary of section names and their corresponding content.
    """
    sections = {}
    current_section = "overview"
    lines = compiled_text.splitlines()
    current_content = []

    for line in lines:
        if line.strip().startswith("## "):
            sections[current_section] = "\n".join(current_content).strip()
            header_name = line.replace("##", "").strip().lower()
            current_section = re.sub(r'[^a-z0-9_]', '_', header_name)
            current_content = []
        else:
            current_content.append(line)

    sections[current_section] = "\n".join(current_content).strip()

    return sections

def extract_final_prompt(compiled_text: str) -> str:
    """
    Extracts the content under '## FINAL OPTIMIZED PROMPT' from the compiled output.
    If not found, returns the entire compiled text.
    """
    # Regex search for FINAL OPTIMIZED PROMPT heading (case-insensitive) and capture everything after
    pattern = r"##\s*FINAL\s*OPTIMIZED\s*PROMPT\s*(.*)"
    match = re.search(pattern, compiled_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return compiled_text

