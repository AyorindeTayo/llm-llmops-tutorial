"""Cost estimation utilities — importable for tests."""


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4


def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    pricing = {
        "gpt-4o":       {"input": 5.0,  "output": 15.0},
        "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku":    {"input": 0.25, "output": 1.25},
    }
    rates = pricing.get(model, {"input": 5.0, "output": 15.0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return round(cost, 6)
