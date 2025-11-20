import unittest
from functions.utils.language_tone import describe_language, describe_tone


class TestLanguageToneUtils(unittest.TestCase):

    def test_describe_language_known(self):
        self.assertEqual(describe_language("en"), "English")
        self.assertEqual(describe_language("th"), "Thai")

    def test_describe_language_unknown(self):
        result = describe_language("xx")
        self.assertIn("code='xx'", result)

    def test_describe_tone_known(self):
        self.assertIn("formal", describe_tone("formal"))
        self.assertIn("academic", describe_tone("academic"))

    def test_describe_tone_unknown(self):
        result = describe_tone("ultra-formal")
        self.assertIn("neutral", result)


if __name__ == "__main__":
    unittest.main()
