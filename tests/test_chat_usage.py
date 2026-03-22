import unittest

from app.services.grok.utils.response import make_chat_response
from app.services.grok.utils.usage import (
    empty_chat_usage,
    estimate_chat_usage,
    estimate_text_tokens,
)


class ChatUsageTests(unittest.TestCase):
    def test_estimate_text_tokens_is_positive(self):
        self.assertGreater(estimate_text_tokens("Reply with exactly: ok"), 0)
        self.assertGreater(estimate_text_tokens("你好，世界"), 0)

    def test_estimate_chat_usage_returns_non_zero_counts(self):
        usage = estimate_chat_usage(
            prompt_text="Reply with exactly: ok",
            completion_text="ok",
        )

        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )
        self.assertEqual(usage["input_tokens"], usage["prompt_tokens"])
        self.assertEqual(usage["output_tokens"], usage["completion_tokens"])

    def test_estimate_chat_usage_counts_tool_calls(self):
        usage = estimate_chat_usage(
            prompt_text="Use a tool",
            completion_tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"weather\"}"},
                }
            ],
        )

        self.assertGreater(usage["completion_tokens"], 0)

    def test_make_chat_response_uses_empty_usage_by_default(self):
        response = make_chat_response("grok-4.20-beta", "ok")
        self.assertEqual(response["usage"], empty_chat_usage())


if __name__ == "__main__":
    unittest.main()
