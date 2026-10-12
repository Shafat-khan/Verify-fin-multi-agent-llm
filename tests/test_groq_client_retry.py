"""
Tests for GroqClient's rate-limit retry-with-backoff logic (fixed
after a real failure: the daily token cap collapsed an entire
evaluation run because 429 errors were never retried).
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from verifyfin.llm.groq_client import GroqClient, _parse_retry_after_seconds


class TestParseRetryAfterSeconds:
    def test_parses_seconds_only_format(self):
        assert _parse_retry_after_seconds("Please try again in 3.2325s.") == pytest.approx(3.2325)

    def test_parses_minutes_and_seconds_format(self):
        assert _parse_retry_after_seconds("Please try again in 8m26.736s.") == pytest.approx(506.736)

    def test_parses_double_digit_minutes(self):
        assert _parse_retry_after_seconds("Please try again in 10m13.872s.") == pytest.approx(613.872)

    def test_no_match_returns_none(self):
        assert _parse_retry_after_seconds("Some unrelated error message.") is None


def _mock_response(content="Response."):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 20
    return mock_response


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    with patch("verifyfin.llm.groq_client.Groq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_groq_cls.return_value = mock_instance
        c = GroqClient()
        c.cache_dir = tmp_path / "cache"
        c.cache_dir.mkdir()
        yield c, mock_instance


class TestRetryWithBackoff:
    def test_succeeds_immediately_when_no_error(self, client):
        c, mock_instance = client
        mock_instance.chat.completions.create.return_value = _mock_response("OK")

        result = c.complete("prompt")

        assert result == "OK"
        assert mock_instance.chat.completions.create.call_count == 1

    def test_retries_after_rate_limit_error_and_succeeds(self, client, monkeypatch):
        c, mock_instance = client
        monkeypatch.setattr(time, "sleep", lambda s: None)  # don't actually wait in tests

        mock_instance.chat.completions.create.side_effect = [
            Exception("Rate limit reached. Please try again in 1.5s."),
            _mock_response("Recovered"),
        ]

        result = c.complete("prompt")

        assert result == "Recovered"
        assert mock_instance.chat.completions.create.call_count == 2

    def test_gives_up_after_max_retries(self, client, monkeypatch):
        c, mock_instance = client
        monkeypatch.setattr(time, "sleep", lambda s: None)

        mock_instance.chat.completions.create.side_effect = Exception(
            "Rate limit reached. Please try again in 1.0s."
        )

        with pytest.raises(Exception, match="Rate limit"):
            c.complete("prompt")

        # 1 initial attempt + 3 retries = 4 total calls
        assert mock_instance.chat.completions.create.call_count == 4

    def test_non_rate_limit_error_is_not_retried(self, client):
        c, mock_instance = client
        mock_instance.chat.completions.create.side_effect = Exception("Some other unrelated failure")

        with pytest.raises(Exception, match="unrelated"):
            c.complete("prompt")

        assert mock_instance.chat.completions.create.call_count == 1  # no retry attempted

    def test_wait_time_is_capped_at_safety_limit(self, client, monkeypatch):
        c, mock_instance = client
        sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

        mock_instance.chat.completions.create.side_effect = [
            Exception("Please try again in 30m0.0s."),  # absurdly long, must be capped
            _mock_response("Recovered"),
        ]

        c.complete("prompt")

        assert sleep_calls[0] <= 900  # the 15-minute safety cap


class TestCacheKeyIncludesMaxTokens:
    def test_different_max_tokens_is_a_cache_miss(self, client):
        """
        Regression test for a real bug: raising max_tokens for a call
        site (e.g. the claim-extraction call, to fix truncated JSON
        responses) did NOT take effect on re-runs, because the cache
        key was identical to the prior, lower-max_tokens call -- so a
        stale, truncated/empty cached response kept being served
        regardless of the code fix. max_tokens is now part of the
        cache key, so a genuinely different token budget is treated as
        a genuinely different request.
        """
        c, mock_instance = client
        mock_instance.chat.completions.create.return_value = _mock_response("OK")
        c.complete("Same prompt", max_tokens=500)
        c.complete("Same prompt", max_tokens=3000)
        assert mock_instance.chat.completions.create.call_count == 2

    def test_same_max_tokens_still_hits_cache(self, client):
        c, mock_instance = client
        mock_instance.chat.completions.create.return_value = _mock_response("OK")
        c.complete("Same prompt", max_tokens=500)
        c.complete("Same prompt", max_tokens=500)
        assert mock_instance.chat.completions.create.call_count == 1
