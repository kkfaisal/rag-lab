"""USD per million tokens. Cache writes are 1.25x input, cache reads are listed
separately. Update when prices change; unknown models cost 0 and print a warning."""

PRICES = {
    #                    input  output  cache_read
    "claude-opus-5-5":   (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5":  (0.10, 0.50, 0.01),
}

_warned: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    if model not in PRICES:
        if model not in _warned:
            print(f"    [pricing] no price for {model}; cost recorded as 0")
            _warned.add(model)
        return 0.0
    inp, out, read = PRICES[model]
    return (input_tokens * inp + output_tokens * out + cache_read * read + cache_write * inp * 1.25) / 1e6
