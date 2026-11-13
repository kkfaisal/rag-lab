"""Anthropic (Claude) provider using the official `anthropic` SDK.

The API key comes from ANTHROPIC_API_KEY (loaded from .env by the CLI).
"""

import anthropic

from .base import LLMProvider, LLMResult, Usage


class AnthropicProvider(LLMProvider):
    def __init__(self, refusal_fallback: bool = True):
        self.client = anthropic.Anthropic(max_retries=5)
        # If a safety classifier declines a request, let the API retry it on a
        # fallback model inside the same call (server-side fallback, beta).
        self.refusal_fallback = refusal_fallback

    def complete(self, *, model, system, user, max_tokens, json_schema=None, effort=None) -> LLMResult:
        output_config = {}
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if effort:
            output_config["effort"] = effort
        kwargs = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if output_config:
            kwargs["output_config"] = output_config
        if self.refusal_fallback:
            resp = self.client.beta.messages.create(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            resp = self.client.messages.create(**kwargs)

        usage = Usage()
        u = resp.usage
        usage.add(resp.model, u.input_tokens, u.output_tokens,
                  u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(text=text, stop_reason=resp.stop_reason, model=resp.model, usage=usage)
