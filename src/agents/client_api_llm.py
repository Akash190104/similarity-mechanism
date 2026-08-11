"""Hosted-model :class:`LLM` implementation backed by OpenAI-compatible APIs."""

import json
import time
from typing import Any

import httpx
from openai import OpenAI, OpenAIError

from config import settings
from src.agents.base import LLM


class ClientAPILLM(LLM):
    """Thin wrapper around hosted LLM APIs (OpenAI-compatible clients)."""

    def __init__(self, model_name: str, provider: str):
        self.provider = provider
        self.model_name = model_name
        self.client = self._get_client()

    def _get_client(self) -> OpenAI:
        """Initialise an :class:`OpenAI` client for the configured provider."""
        match self.provider:
            case "OpenAI":
                return OpenAI(
                    api_key=settings.OPENAI_API_KEY,
                    # Retry/timeout config
                    max_retries=3,
                    timeout=600,
                )
            case "Gemini":
                return OpenAI(
                    api_key=settings.GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=3,
                    timeout=600,
                )
            case "OpenRouter":
                return OpenAI(
                    api_key=settings.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=3,
                    timeout=600,
                )
            case _:
                raise ValueError(f"Unknown provider {self.provider}")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Call the remote chat completion endpoint with basic retry/backoff.

        Content-filter / safety refusals are not retried — they are deterministic
        for a given prompt, so retrying just burns time. We return an empty
        string in that case (and on terminal exhaustion of transient retries),
        letting the caller's parser fall back to a default answer rather than
        propagating an exception that would kill the entire benchmark run.
        """
        extra_body = {}
        if "reasoning" in kwargs:
            extra_body["reasoning"] = kwargs.pop("reasoning")
        if "provider" in kwargs:
            extra_body["provider"] = kwargs.pop("provider")

        # Short retry budget: 3 attempts with delays [0, 2, 4] = 6s of sleep.
        delays = [2, 4]
        for attempt, delay in enumerate([0] + delays):
            if delay:
                time.sleep(delay)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body=extra_body if extra_body else None,
                    **kwargs,
                )
                choice = completion.choices[0]

                # Content-filter / safety refusal — don't retry, return empty.
                finish = getattr(choice, "finish_reason", None)
                if finish in ("content_filter", "safety"):
                    print(f"Content filter triggered ({self.model_name}); skipping.")
                    return ""

                if hasattr(choice, "error") and choice.error:
                    error_msg = choice.error.get("message", "Unknown error")
                    error_code = choice.error.get("code", "unknown")
                    if _is_content_filter(str(error_code), str(error_msg)):
                        print(f"Content filter ({self.model_name}, code {error_code}); skipping.")
                        return ""
                    raise OpenAIError(f"API error (code {error_code}): {error_msg}")

                content = choice.message.content
                if not content or content.strip() == "":
                    native_finish = getattr(choice, "native_finish_reason", None)
                    usage = getattr(completion, "usage", None)
                    print(
                        f"Empty content from {self.model_name}: "
                        f"finish_reason={finish!r} native_finish_reason={native_finish!r} "
                        f"usage={usage!r}",
                        flush=True,
                    )
                    # A ``length`` finish means reasoning consumed the whole token
                    # budget without emitting an answer — deterministic for this
                    # prompt, so retrying just burns the budget again. Fail fast.
                    if finish == "length":
                        return ""
                    raise OpenAIError(
                        f"API returned empty response content (finish_reason={finish})"
                    )
                return content
            except Exception as e:
                if _is_content_filter("", str(e)):
                    print(f"Content filter ({self.model_name}): {e}; skipping.")
                    return ""
                if attempt == len(delays):
                    print(f"API call failed terminally after {attempt + 1} attempts: "
                          f"{type(e).__name__}: {e}; returning empty.")
                    return ""
                print(f"API call failed (attempt {attempt + 1}/{len(delays) + 1}): "
                      f"{type(e).__name__}: {e}")
                continue
        return ""


_CONTENT_FILTER_HINTS = (
    "content_filter", "content filter", "moderation", "safety",
    "blocked", "harm", "unsafe content",
)


def _is_content_filter(code: str, message: str) -> bool:
    text = f"{code} {message}".lower()
    return any(h in text for h in _CONTENT_FILTER_HINTS)
