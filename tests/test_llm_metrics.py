# tests/test_llm_metrics.py
"""
Unittest suite for utils/llm_metrics.py

Covers:
- pricing lookup (model-specific + default)
- usage token extraction across SDK shapes (OpenAI + Gemini, object + dict + wrapped)
- text extraction
- end-to-end call wrapper: logs via structlog + writes CSV
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import datetime as dt  # for building real datetime objects for patches

from functions.utils import llm_metrics


# ----------------------------
# Fake responses (SDK shapes)
# ----------------------------

class FakeOpenAIUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class FakeOpenAIChoice:
    def __init__(self, content: str):
        self.message = {"content": content}

class FakeOpenAIResponse:
    def __init__(self, text: str, pt: int, ct: int):
        self.usage = FakeOpenAIUsage(pt, ct)
        self.choices = [FakeOpenAIChoice(text)]

class FakeGeminiResponse:
    """Simulate Gemini-like response exposing .text and .usage as dict."""
    def __init__(self, text: str, pt: int, ct: int):
        self.text = text
        self.usage = {"prompt_tokens": pt, "completion_tokens": ct}

# --- Gemini object-style shapes ---

from types import SimpleNamespace

class FakeGeminiUsageMeta:
    def __init__(
        self,
        prompt_token_count: int | None = None,
        candidates_token_count: int | None = None,
        total_token_count: int | None = None,
    ):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count

class FakeGeminiCandidate:
    def __init__(self, finish_reason_name: str | None = None):
        # Simulate Enum-like .name for finish reason
        self.finish_reason = SimpleNamespace(name=finish_reason_name) if finish_reason_name else None

class FakeGeminiResponseObj:
    """Gemini-like: .text, .usage_metadata (object), .candidates (finish_reason)"""
    def __init__(self, text: str, pt: int, ct: int, finish_reason_name: str | None = None):
        self.text = text
        self.usage_metadata = FakeGeminiUsageMeta(pt, ct, pt + ct)
        self.candidates = [FakeGeminiCandidate(finish_reason_name)]


# ----------------------------
# Tests
# ----------------------------

class TestLLMMetrics(unittest.TestCase):
    """Test suite for utils/llm_metrics.py"""

    def setUp(self):
        # Clear caches now that we read csv path from YAML
        llm_metrics._load_config.cache_clear()
        llm_metrics._load_pricing.cache_clear()
        llm_metrics._get_csv_path.cache_clear()

        # temp CSV path
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmpdir.name, "llm_call_logs.csv")

        # Patch _get_csv_path() to return our temp file
        self._csv_patch = patch("functions.utils.llm_metrics._get_csv_path", return_value=self.csv_path)
        self._csv_patch.start()

        # Pretty header
        test_name = self._testMethodName
        print("\n" + "=" * 90, file=sys.stderr)
        print(f"🧪 STARTING TEST: {test_name}", file=sys.stderr)
        print("=" * 90, file=sys.stderr)

    def tearDown(self):
        print("-" * 90 + "\n", file=sys.stderr)
        self._csv_patch.stop()
        self.tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Token & text extraction
    # ------------------------------------------------------------------

    def test_extract_usage_tokens_gemini_usage_metadata(self):
        gem_obj = FakeGeminiResponseObj("hello", pt=11, ct=7, finish_reason_name="STOP")
        self.assertEqual(llm_metrics.extract_usage_tokens(gem_obj), (11, 7))

    def test_extract_usage_tokens_variants(self):
        openai_resp = FakeOpenAIResponse("hi", pt=123, ct=45)
        gemini_resp = FakeGeminiResponse("hello", pt=10, ct=5)
        dict_resp = {"usage": {"input_tokens": 7, "output_tokens": 3}, "text": "ok"}

        self.assertEqual(llm_metrics.extract_usage_tokens(openai_resp), (123, 45))
        self.assertEqual(llm_metrics.extract_usage_tokens(gemini_resp), (10, 5))
        self.assertEqual(llm_metrics.extract_usage_tokens(dict_resp), (7, 3))

    def test_extract_text_variants(self):
        openai_resp = FakeOpenAIResponse("openai-text", 1, 1)
        gemini_resp = FakeGeminiResponse("gemini-text", 1, 1)
        dict_resp = {"usage": {}, "text": "dict-text"}

        self.assertEqual(llm_metrics.extract_text(openai_resp), "openai-text")
        self.assertEqual(llm_metrics.extract_text(gemini_resp), "gemini-text")
        self.assertEqual(llm_metrics.extract_text(dict_resp), "dict-text")

    # ---- New Gemini edge cases ----

    def test_extract_usage_tokens_gemini_prompt_plus_total_only(self):
        """
        Some Gemini responses omit candidates_token_count but include total_token_count.
        We should derive output = total - prompt.
        """
        class Resp:
            pass
        r = Resp()
        r.text = "derived"
        r.usage_metadata = FakeGeminiUsageMeta(prompt_token_count=21, candidates_token_count=None, total_token_count=50)
        self.assertEqual(llm_metrics.extract_usage_tokens(r), (21, 29))

    def test_extract_usage_tokens_gemini_wrapped_in_response(self):
        """
        Some clients wrap the SDK response under resp.response.
        """
        inner = FakeGeminiResponseObj("wrapped", pt=13, ct=8, finish_reason_name="STOP")
        resp = {"response": inner}
        self.assertEqual(llm_metrics.extract_usage_tokens(resp), (13, 8))

    def test_extract_usage_tokens_proxy_top_level_totals(self):
        """
        Proxies may expose token counts at top-level dict using *_token_count keys.
        """
        resp = {
            "text": "proxy",
            "prompt_token_count": 30,
            "total_token_count": 44,   # no candidates_token_count → derive 14
        }
        self.assertEqual(llm_metrics.extract_usage_tokens(resp), (30, 14))

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def test_get_rates_model_specific_and_default(self):
        pricing = {
            "default": {"usd_per_input_token": 1e-6, "usd_per_output_token": 3e-6},
            "gemini-2.5-flash": {"usd_per_input_token": 1.25e-6, "usd_per_output_token": 5e-6},
        }
        with patch.object(llm_metrics, "_load_pricing", return_value=pricing):
            in_rate, out_rate = llm_metrics._get_rates("gemini-2.5-flash")
            self.assertAlmostEqual(in_rate, 1.25e-6)
            self.assertAlmostEqual(out_rate, 5e-6)

            in_def, out_def = llm_metrics._get_rates("unknown-model")
            self.assertAlmostEqual(in_def, 1e-6)
            self.assertAlmostEqual(out_def, 3e-6)

    # ------------------------------------------------------------------
    # End-to-end wrapper: CSV + structlog line
    # ------------------------------------------------------------------

    def test_call_wrapper_logs_and_writes_csv_openai(self):
        # Simple pricing for predictable math
        pricing = {"default": {"usd_per_input_token": 2e-6, "usd_per_output_token": 6e-6}}
        with patch.object(llm_metrics, "_load_pricing", return_value=pricing):
            # Deterministic response time (123 ms)
            with patch("functions.utils.llm_metrics.time.monotonic", side_effect=[100.0, 100.123]):
                # Deterministic CSV timestamp — patch module's datetime.datetime.now(...)
                with patch("functions.utils.llm_metrics.datetime") as mock_dt:
                    mock_dt.datetime.now.return_value = dt.datetime(2025, 11, 12, 7, 0, 0, tzinfo=dt.timezone.utc)
                    mock_dt.UTC = dt.timezone.utc

                    client = MagicMock()
                    client.generate.return_value = FakeOpenAIResponse("wrapped-text", pt=1000, ct=400)

                    text = llm_metrics.call_llm_section_with_metrics(
                        llm_client=client,
                        model="any-model",
                        prompt="Hello",
                        section_id="profile_summary",
                        purpose="stage_b_generation",
                        user_id="user-123",
                    )

        # Return text
        self.assertEqual(text, "wrapped-text")

        # CSV exists with one row
        self.assertTrue(os.path.isfile(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["user_id"], "user-123")
        self.assertEqual(row["model"], "any-model")
        self.assertEqual(row["section_id"], "profile_summary")
        self.assertEqual(row["purpose"], "stage_b_generation")
        self.assertEqual(int(row["input_tokens"]), 1000)
        self.assertEqual(int(row["output_tokens"]), 400)
        # Cost = 1000*2e-6 + 400*6e-6 = 0.002 + 0.0024 = 0.0044 USD
        self.assertAlmostEqual(float(row["input_cost_usd"]), 0.002, places=6)
        self.assertAlmostEqual(float(row["output_cost_usd"]), 0.0024, places=6)
        self.assertAlmostEqual(float(row["total_cost_usd"]), 0.0044, places=6)
        self.assertEqual(int(row["response_time_ms"]), 123)

    def test_call_wrapper_logs_and_writes_csv_gemini(self):
        pricing = {"default": {"usd_per_input_token": 1e-6, "usd_per_output_token": 2e-6}}
        with patch.object(llm_metrics, "_load_pricing", return_value=pricing):
            # deterministic 123ms
            with patch("functions.utils.llm_metrics.time.monotonic", side_effect=[200.0, 200.123]):
                # deterministic timestamp
                with patch("functions.utils.llm_metrics.datetime") as mock_dt:
                    mock_dt.datetime.now.return_value = dt.datetime(2025, 11, 12, 7, 0, 0, tzinfo=dt.timezone.utc)
                    mock_dt.UTC = dt.timezone.utc

                    client = MagicMock()
                    client.generate.return_value = FakeGeminiResponseObj(
                        text="gemini-wrapped",
                        pt=100,
                        ct=40,
                        finish_reason_name="STOP",
                    )

                    text = llm_metrics.call_llm_section_with_metrics(
                        llm_client=client,
                        model="gemini-2.5-flash",
                        prompt="Hi",
                        section_id="skills",
                        purpose="stage_b_generation",
                        user_id="user-456",
                    )

        self.assertEqual(text, "gemini-wrapped")
        self.assertTrue(os.path.isfile(self.csv_path))
        with open(self.csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["user_id"], "user-456")
        self.assertEqual(row["model"], "gemini-2.5-flash")
        self.assertEqual(row["section_id"], "skills")
        self.assertEqual(row["purpose"], "stage_b_generation")
        self.assertEqual(int(row["input_tokens"]), 100)
        self.assertEqual(int(row["output_tokens"]), 40)
        # cost: 100*1e-6 + 40*2e-6 = 0.0001 + 0.00008 = 0.00018
        self.assertAlmostEqual(float(row["input_cost_usd"]), 0.0001, places=8)
        self.assertAlmostEqual(float(row["output_cost_usd"]), 0.00008, places=8)
        self.assertAlmostEqual(float(row["total_cost_usd"]), 0.00018, places=8)
        self.assertEqual(int(row["response_time_ms"]), 123)


if __name__ == "__main__":
    unittest.main()
