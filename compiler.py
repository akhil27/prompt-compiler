from llm import generate, DEFAULT_MODEL
from prompts import COMPILER_SYSTEM_PROMPT


def compile_prompt(user_input: str, model: str = DEFAULT_MODEL, temperature: float = 0.3) -> str:
    """
    Takes a raw, vague user prompt and compiles it into an optimized,
    highly structured prompt template using the Hugging Face client.

    Parameters:
        user_input (str): The raw instruction/prompt from the user.
        model (str): The Hugging Face model name.
        temperature (float): Controls response creativity.

    Returns:
        str: The compiled, structured prompt.
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input prompt cannot be empty.")

    prompt_payload = (
        f"Please compile the following raw prompt into a structured instruction template:\n\n"
        f"\"\"\"\n{user_input.strip()}\n\"\"\""
    )

    compiled_result = generate(
        prompt=prompt_payload,
        system_prompt=COMPILER_SYSTEM_PROMPT,
        model=model,
        temperature=temperature
    )

    return compiled_result.strip()
