import json
import os
from typing import Any


def _maybe_load_dotenv() -> None:
    """
    Load key/value pairs from a local .env into process env.
    This is optional: if python-dotenv isn't installed, we just skip.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def _mask_secret(value: str) -> str:
    v = value.strip()
    if len(v) <= 8:
        return "***"
    return v[:4] + "..." + v[-4:]


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove leading ```lang? and trailing ```
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.endswith("```"):
            t = t[: -len("```")]
    return t.strip()


def _coerce_tasks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        comp = item.get("compensation")
        if not isinstance(task, str) or not task.strip():
            continue
        try:
            comp_f = float(comp)
        except Exception:
            comp_f = 0.0
        out.append({"task": task.strip(), "compensation": comp_f})
    return out


def extract_human_tasks_from_prompt(prompt: str) -> list[dict[str, Any]]:
    _maybe_load_dotenv()
    """
    Extract a list of *human-doable* tasks from a prompt and assign USD compensation.

    Returns:
        [{"task": str, "compensation": float}, ...]
    """
    api_key = os.getenv("CORALFLAVOR_API_KEY") or os.getenv("CORAL_API_KEY")
    if not api_key:
        debug = os.getenv("EXTRACTLABOR_DEBUG", "").lower() in {"1", "true", "yes"}
        if debug:
            print(
                "[extractlabor] Missing CORALFLAVOR_API_KEY (or CORAL_API_KEY); returning []"
            )
        # Keep dev flows running even without credentials.
        return []

    # Lazy import so the repo can still be imported without the dependency installed.
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("CORALFLAVOR_BASE_URL", "https://coralflavor.com/v1"),
        api_key=api_key,
    )
    model = os.environ.get("CORALFLAVOR_MODEL", "Coralflavor")

    system = (
        "You extract tasks that require real-world human work from user prompts.\n"
        "Return ONLY valid JSON (no markdown) as an array of objects with keys:\n"
        '- "task": string (clear, actionable, human-only)\n'
        '- "compensation": number (USD, realistic)\n'
        "If there are no human-only tasks, return []."
    )

    debug = os.getenv("EXTRACTLABOR_DEBUG", "").lower() in {"1", "true", "yes"}
    user = system + f"\n\nPrompt:\n{prompt}\n\nReturn JSON now:"
    if debug:
        print(f"[extractlabor] Using model={model} base_url={os.getenv('CORALFLAVOR_BASE_URL', 'https://coralflavor.com/v1')}")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        # Common failure modes: quota/rate-limit (429), invalid key (401), network issues.
        if debug:
            print(
                f"[extractlabor] Coralflavor error using model={model}, key={_mask_secret(api_key)}: {e}"
            )
        return []

    raw = (
        _strip_code_fences(resp.choices[0].message.content)
        if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content
        else ""
    )
    if not raw:
        if debug:
            print("[extractlabor] Empty model response text; returning []")
        return []

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage if the model wrapped JSON in extra text.
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except Exception:
                return []
        else:
            return []

    tasks = _coerce_tasks(parsed)
    if debug:
        print(f"[extractlabor] Parsed tasks: {tasks}")
        print(tasks)
    return tasks