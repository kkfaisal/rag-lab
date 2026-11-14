"""AI help in the code editor: Claude rewrites a script node's code to follow an
instruction ("chunk by markdown headings", "fix this error", "add docstrings").

It only proposes code; the editor shows the change as a diff and the user decides
whether to accept it. Needs ANTHROPIC_API_KEY. The model is claude-sonnet-5-5 unless
RAG_LAB_ASSIST_MODEL says otherwise.
"""

import json
import os

from . import scripting
from .llm import get_provider

SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "The complete new script"},
        "explanation": {"type": "string", "description": "What changed and why, in a few sentences"},
    },
    "required": ["code", "explanation"],
    "additionalProperties": False,
}

SYSTEM = """You edit Python scripts for rag-lab, a playground for building and benchmarking
retrieval-augmented generation (RAG) pipelines. Each script plugs into one pipeline
stage and must keep that stage's contract: the function names and signatures that
rag-lab calls, and the shapes of what they return. The contract is given as a
reference template with docstrings.

Return the complete new script, not a fragment. Keep changes focused on the
instruction, keep the user's existing style and constants unless asked otherwise,
and only use libraries that are already imported or are in the Python standard
library, numpy, sentence-transformers, or torch. If the instruction can't be done
within the contract, keep the code unchanged and say why in the explanation."""


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def suggest(kind: str, code: str, instruction: str, error: str | None = None) -> dict:
    contract = scripting.CONTRACTS.get(kind)
    parts = []
    if contract:
        parts.append(f"<stage>{kind}</stage>\n<contract>\nrag-lab calls: {', '.join(contract['required'])}"
                     f"{' (optional: ' + ', '.join(contract['optional']) + ')' if contract['optional'] else ''}\n"
                     f"Reference template:\n{contract['template']}</contract>")
    parts.append(f"<current_script>\n{code}\n</current_script>")
    if error:
        parts.append(f"<last_error>\n{error}\n</last_error>")
    parts.append(f"<instruction>\n{instruction}\n</instruction>")
    model = os.environ.get("RAG_LAB_ASSIST_MODEL", "claude-sonnet-5-5")
    res = get_provider("anthropic").complete(model=model, system=SYSTEM, user="\n\n".join(parts),
                                             max_tokens=16000, json_schema=SCHEMA, effort="medium")
    if res.stop_reason == "max_tokens":
        raise RuntimeError("The answer was cut off (script too long). Try a narrower instruction.")
    out = json.loads(res.text)
    return {**out, "model": res.model, "cost_usd": res.usage.cost_usd}
