import unittest

from functions.utils.common import resolve_token_budget  # or whatever name you pick


class TestResolveTokenBudget(unittest.TestCase):
    def test_scalar_budget(self):
        params = {
            "section_token_budgets": {
                "profile_summary": 150,
                "default": 50,
            }
        }
        self.assertEqual(
            resolve_token_budget("profile_summary", attempt=1, params=params),
            150,
        )
        # scalar → same for any attempt
        self.assertEqual(
            resolve_token_budget("profile_summary", attempt=3, params=params),
            150,
        )

    def test_list_budget_clamps_to_last(self):
        params = {
            "section_token_budgets": {
                "profile_summary": [100, 200],
                "default": 50,
            }
        }
        self.assertEqual(
            resolve_token_budget("profile_summary", attempt=1, params=params),
            100,
        )
        self.assertEqual(
            resolve_token_budget("profile_summary", attempt=2, params=params),
            200,
        )
        # attempt > len(list) → clamp to last
        self.assertEqual(
            resolve_token_budget("profile_summary", attempt=5, params=params),
            200,
        )

    def test_falls_back_to_default_and_handles_missing(self):
        params = {
            "section_token_budgets": {
                "default": 60,
            }
        }
        # no explicit entry → use default
        self.assertEqual(
            resolve_token_budget("experience", attempt=1, params=params),
            60,
        )

        # no section_token_budgets at all → None
        self.assertIsNone(
            resolve_token_budget("experience", attempt=1, params={}),
        )

    def test_invalid_or_non_positive_values_return_none(self):
        params = {
            "section_token_budgets": {
                "profile_summary": [0, -10],
                "default": -5,
            }
        }
        self.assertIsNone(
            resolve_token_budget("profile_summary", attempt=1, params=params),
        )
        self.assertIsNone(
            resolve_token_budget("other", attempt=1, params=params),
        )
