"""The LLM provider interface used by the generate stage."""

from dataclasses import asdict, dataclass, field

from .pricing import cost_usd


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0       # tokens served from the prompt cache (cheap)
    cache_creation_input_tokens: int = 0   # tokens written to the prompt cache (slightly pricier)
    requests: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0) -> None:
        c = cost_usd(model, input_tokens, output_tokens, cache_read, cache_write)
        for target in (self, self.by_model.setdefault(model, Usage())):
            target.input_tokens += input_tokens
            target.output_tokens += output_tokens
            target.cache_read_input_tokens += cache_read
            target.cache_creation_input_tokens += cache_write
            target.requests += 1
            target.cost_usd = round(target.cost_usd + c, 6)

    def merge(self, other: "Usage") -> None:
        for model, u in other.by_model.items():
            self.add(model, u.input_tokens, u.output_tokens, u.cache_read_input_tokens, u.cache_creation_input_tokens)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["by_model"] = {m: {k: v for k, v in asdict(u).items() if k != "by_model"} for m, u in self.by_model.items()}
        return d


@dataclass
class LLMResult:
    text: str
    stop_reason: str | None
    model: str
    usage: Usage


class LLMProvider:
    """One method: send a system prompt + user message, optionally constrained to a
    JSON schema, and get text + usage back."""

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 json_schema: dict | None = None, effort: str | None = None) -> LLMResult:
        raise NotImplementedError


def get_provider(name: str = "anthropic", **kwargs) -> LLMProvider:
    if name == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(**kwargs)
    raise SystemExit(f"Unknown LLM provider {name!r}. Available: anthropic")
