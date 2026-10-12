"""
verifyfin.llm.groq_client
============================
Rate-limited, disk-cached wrapper around the Groq API.

Rate limiting: a sliding 60-second window of call timestamps handles
Groq's PER-MINUTE limit proactively.

Retry-with-backoff (added after a real failure found during the
evaluation harness's first full run): Groq also enforces a DAILY token
cap (TPD) separate from the per-minute limit. The proactive rate
limiter above has no way to anticipate this -- it only tracks calls
made in the last 60 seconds -- so once the daily cap was hit partway
through the evaluation, every subsequent call failed immediately with
a 429 error suggesting a 5-10 minute wait, and since nothing caught or
retried this, entire scenarios silently failed for the rest of the
run (this is why JNJ ended up with zero completed VERIFY-FIN scenarios
in that run). complete() now catches a 429-style error, parses Groq's
own "try again in Xm Ys" hint from the message, sleeps that long (up
to a safety cap), and retries -- so a long-running evaluation can
recover from a daily-cap wall instead of losing an entire condition.

Disk caching: every (model, system_prompt, user_prompt, temperature,
cache_namespace) combination is hashed to a cache key. cache_namespace
keeps different evaluation conditions' costs measured independently
(see prior module history) -- two GroqClient instances with no
namespace behave exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

from groq import Groq

from verifyfin.utils.io_utils import load_config
from verifyfin.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MAX_RATE_LIMIT_RETRIES = 3
_MAX_RATE_LIMIT_WAIT_SECONDS = 900  # 15-minute safety cap -- if Groq asks for longer, we give up rather than hang
_RETRY_AFTER_PATTERN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _parse_retry_after_seconds(error_message: str) -> Optional[float]:
    """
    Parses Groq's own suggested wait time from a rate-limit error
    message, e.g. "Please try again in 3.2325s." or "...in 8m26.736s.".
    Returns None if the message doesn't match the expected format --
    callers should treat that as "don't know how long to wait".
    """
    match = _RETRY_AFTER_PATTERN.search(error_message)
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


class GroqRateLimiter:
    def __init__(self, max_requests_per_minute: int):
        self.max_requests_per_minute = max_requests_per_minute
        self._timestamps: deque[float] = deque()

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_requests_per_minute:
            oldest = self._timestamps[0]
            sleep_for = 60 - (now - oldest) + 0.1
            if sleep_for > 0:
                logger.info(
                    f"Rate limit reached ({self.max_requests_per_minute}/min). "
                    f"Sleeping {sleep_for:.1f}s before next call."
                )
                time.sleep(sleep_for)

        self._timestamps.append(time.monotonic())


class GroqClient:
    def __init__(self, config_path: Optional[str] = None, cache_namespace: Optional[str] = None):
        self.config = load_config(config_path)
        llm_cfg = self.config["llm"]

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY is not set or is still the placeholder value. "
                "Copy .env.example to .env and add your real key "
                "(get one free at https://console.groq.com/keys)."
            )

        self._client = Groq(api_key=api_key)
        self.model = llm_cfg["model"]
        self.default_temperature = llm_cfg["temperature"]
        self.default_max_tokens = llm_cfg["max_tokens"]

        self.cache_enabled = llm_cfg.get("cache_enabled", True)
        self.cache_dir = Path(llm_cfg.get("cache_dir", "./data/raw/llm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_namespace = cache_namespace

        self._rate_limiter = GroqRateLimiter(llm_cfg.get("rate_limit_per_minute", 28))

        self.call_count = 0
        self.cache_hit_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        logger.info(
            f"GroqClient initialised (model={self.model}, "
            f"cache_enabled={self.cache_enabled}, "
            f"cache_namespace={cache_namespace!r}, "
            f"rate_limit={llm_cfg.get('rate_limit_per_minute', 28)}/min)"
        )

    def _cache_key(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """
        max_tokens is included in the cache key (fixed after a real bug:
        raising max_tokens for the claim-extraction call site did not
        actually take effect on a re-run, because the cache key was
        identical to the prior, lower-max_tokens call -- so the OLD,
        truncated/empty cached response kept being served regardless of
        the code fix. Any change to the token budget for a given prompt
        must now be treated as a genuinely different request.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "user": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "namespace": self.cache_namespace,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[str]:
        if not self.cache_enabled:
            return None
        path = self._cache_path(key)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            logger.debug(f"Cache hit ({key[:8]}...)")
            return data["response"]
        return None

    def _write_cache(self, key: str, response: str) -> None:
        if not self.cache_enabled:
            return
        with open(self._cache_path(key), "w") as f:
            json.dump({"response": response}, f)

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def get_stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "cache_hit_count": self.cache_hit_count,
            "api_call_count": self.call_count - self.cache_hit_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def reset_stats(self) -> None:
        self.call_count = 0
        self.cache_hit_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "You are a precise financial analysis assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        cache_key = self._cache_key(system_prompt, user_prompt, temp, tokens)
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.call_count += 1
            self.cache_hit_count += 1
            return cached

        self._rate_limiter.wait_if_needed()

        response = self._call_with_retry(system_prompt, user_prompt, temp, tokens, cache_key)
        text = response.choices[0].message.content
        self._write_cache(cache_key, text)

        self.call_count += 1
        self._record_usage(response)

        return text

    def _call_with_retry(self, system_prompt: str, user_prompt: str, temp: float, tokens: int, cache_key: str):
        """
        Calls the Groq API, retrying on rate-limit errors by sleeping
        for the duration Groq itself suggests (parsed from the error
        message), up to _MAX_RATE_LIMIT_RETRIES attempts. This is what
        lets a long-running evaluation survive hitting the DAILY token
        cap partway through, rather than losing the rest of a scenario.
        """
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                logger.info(f"Calling Groq ({self.model}), prompt hash={cache_key[:8]}...")
                return self._client.chat.completions.create(
                    model=self.model,
                    temperature=temp,
                    max_tokens=tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            except Exception as e:
                last_error = e
                wait_seconds = _parse_retry_after_seconds(str(e))
                if wait_seconds is None or attempt == _MAX_RATE_LIMIT_RETRIES:
                    raise
                wait_seconds = min(wait_seconds + 1.0, _MAX_RATE_LIMIT_WAIT_SECONDS)
                logger.warning(
                    f"Groq rate limit hit (attempt {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES}); "
                    f"waiting {wait_seconds:.0f}s before retrying..."
                )
                time.sleep(wait_seconds)
        raise last_error  # unreachable in practice -- satisfies type checkers
