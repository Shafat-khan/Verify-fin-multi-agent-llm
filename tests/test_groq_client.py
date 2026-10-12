"""
Unit tests for verifyfin.llm.groq_client.

The Groq SDK is mocked -- these tests run offline and don't require
curl_cffi to make any real network/TLS calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from verifyfin.llm.groq_client import GroqClient, GroqRateLimiter


def _mock_response(content="Response text.", prompt_tokens=10, completion_tokens=20, total_tokens=30):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.total_tokens = total_tokens
    return mock_response


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-offline-tests")

    with patch("verifyfin.llm.groq_client.Groq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create.return_value = _mock_response()
        mock_groq_cls.return_value = mock_instance

        c = GroqClient()
        c.cache_dir = tmp_path / "cache"
        c.cache_dir.mkdir()

        yield c, mock_instance


class TestGroqClientCaching:
    def test_first_call_hits_the_mocked_api(self, client):
        c, mock_instance = client
        result = c.complete("What was AAPL's Q3 2024 revenue?")
        assert "Response text." in result
        assert mock_instance.chat.completions.create.call_count == 1

    def test_identical_second_call_is_served_from_cache(self, client):
        c, mock_instance = client
        c.complete("What was AAPL's Q3 2024 revenue?")
        c.complete("What was AAPL's Q3 2024 revenue?")
        assert mock_instance.chat.completions.create.call_count == 1

    def test_different_prompts_both_hit_the_api(self, client):
        c, mock_instance = client
        c.complete("Prompt A")
        c.complete("Prompt B")
        assert mock_instance.chat.completions.create.call_count == 2

    def test_different_temperature_is_a_cache_miss(self, client):
        c, mock_instance = client
        c.complete("Same prompt", temperature=0.1)
        c.complete("Same prompt", temperature=0.9)
        assert mock_instance.chat.completions.create.call_count == 2


class TestGroqRateLimiter:
    def test_calls_under_the_limit_do_not_sleep(self):
        import time

        limiter = GroqRateLimiter(max_requests_per_minute=10)
        start = time.monotonic()
        for _ in range(10):
            limiter.wait_if_needed()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0


class TestGroqClientValidation:
    def test_missing_or_placeholder_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqClient()


class TestUsageInstrumentation:
    def test_real_api_call_increments_call_count(self, client):
        c, mock_instance = client
        c.complete("Prompt A")
        stats = c.get_stats()
        assert stats["call_count"] == 1
        assert stats["cache_hit_count"] == 0
        assert stats["api_call_count"] == 1

    def test_cache_hit_increments_call_count_but_not_api_call_count(self, client):
        c, mock_instance = client
        c.complete("Prompt A")
        c.complete("Prompt A")  # cache hit
        stats = c.get_stats()
        assert stats["call_count"] == 2
        assert stats["cache_hit_count"] == 1
        assert stats["api_call_count"] == 1

    def test_token_usage_is_recorded_from_response(self, client):
        c, mock_instance = client
        mock_instance.chat.completions.create.return_value = _mock_response(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        c.complete("Prompt A")
        stats = c.get_stats()
        assert stats["total_prompt_tokens"] == 100
        assert stats["total_completion_tokens"] == 50
        assert stats["total_tokens"] == 150

    def test_token_usage_accumulates_across_calls(self, client):
        c, mock_instance = client
        c.complete("Prompt A")  # 10/20/30 tokens (default mock)
        c.complete("Prompt B")  # another 10/20/30
        stats = c.get_stats()
        assert stats["total_tokens"] == 60

    def test_cached_response_does_not_double_count_tokens(self, client):
        """A cache hit returns instantly with no real API usage -- must not add phantom token counts."""
        c, mock_instance = client
        c.complete("Prompt A")
        c.complete("Prompt A")  # cache hit, no new usage
        stats = c.get_stats()
        assert stats["total_tokens"] == 30  # only the first real call's usage

    def test_missing_usage_data_does_not_crash(self, client):
        c, mock_instance = client
        response_without_usage = MagicMock(spec=["choices"])
        mock_choice = MagicMock()
        mock_choice.message.content = "Response with no usage field."
        response_without_usage.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = response_without_usage

        result = c.complete("Prompt A")  # must not raise
        assert result == "Response with no usage field."

    def test_reset_stats_zeroes_all_counters(self, client):
        c, mock_instance = client
        c.complete("Prompt A")
        c.reset_stats()
        stats = c.get_stats()
        assert stats["call_count"] == 0
        assert stats["total_tokens"] == 0
