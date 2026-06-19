from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


import os
from dotenv import load_dotenv

def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    # 1. Resolve the repo root (current file is in src/, so parent is root)
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # 2. Optionally load values from `.env`
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # 3. Create `state/` if it does not exist
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Choose sensible defaults for compact memory
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Read LLM Provider configurations
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL")

    main_model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temp,
        api_key=api_key,
        base_url=base_url
    )

    # Set up Judge model (defaults to same config or overrides)
    judge_provider = os.getenv("JUDGE_PROVIDER", provider).lower()
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    
    judge_api_key = api_key
    judge_base_url = base_url
    if judge_provider != provider:
        if judge_provider == "openai":
            judge_api_key = os.getenv("OPENAI_API_KEY")
            judge_base_url = os.getenv("OPENAI_BASE_URL")
        elif judge_provider == "gemini":
            judge_api_key = os.getenv("GEMINI_API_KEY")
        elif judge_provider == "anthropic":
            judge_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif judge_provider == "openrouter":
            judge_api_key = os.getenv("OPENROUTER_API_KEY")
            judge_base_url = os.getenv("OPENROUTER_BASE_URL")
        elif judge_provider == "ollama":
            judge_base_url = os.getenv("OLLAMA_BASE_URL")
        elif judge_provider == "custom":
            judge_api_key = os.getenv("CUSTOM_API_KEY")
            judge_base_url = os.getenv("CUSTOM_BASE_URL")

    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=temp,
        api_key=judge_api_key,
        base_url=judge_base_url
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=main_model,
        judge_model=judge_model
    )
