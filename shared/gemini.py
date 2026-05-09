"""Gemini API helpers — backend-agnostic.

Both shared/judging.py (the judge model) and
red_team/trigger_attack/pipeline.py (strategic-error generation during
red-team data prep) need to call Gemini. They both import these.

Why not just google-generativeai directly: rate-limit handling and
the env-var/Colab-Secrets dance are the same in both places, so we
centralise here.

Set GEMINI_API_KEY in your environment before calling setup_gemini().
"""

from __future__ import annotations

import os
import random
import time


def setup_gemini(
    *,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.7,
    max_output_tokens: int = 800,
):
    """Configure the Gemini SDK and return a GenerativeModel.

    Reads GEMINI_API_KEY from the environment, falling back to Colab
    Secrets if running in Colab. Raises ValueError if no key is found.
    """
    import google.generativeai as genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        try:
            from google.colab import userdata  # type: ignore

            api_key = userdata.get("GEMINI_API_KEY")
        except Exception:
            pass
    if api_key is None:
        raise ValueError(
            "Set GEMINI_API_KEY via environment variable or Colab Secrets."
        )

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature":       temperature,
            "max_output_tokens": max_output_tokens,
        },
    )


def gemini_call_with_retry(
    model,
    prompt: str,
    *,
    max_retries: int = 5,
) -> str | None:
    """Call a Gemini model with exponential-backoff retry on rate limits.

    On 429 / "Resource exhausted", waits 2^attempt + jitter seconds and
    retries. On other errors, returns None immediately so the caller
    can decide what to do.

    Returns None if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt).text
        except Exception as e:
            err = str(e)
            if "429" in err or "Resource exhausted" in err:
                wait = (2 ** attempt) + random.random() * 2
                print(f"      Rate limited, waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                print(f"      Gemini error: {e}")
                return None
    print(f"      Failed after {max_retries} retries")
    return None
