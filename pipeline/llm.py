"""One client for every model. Ollama, vLLM, SGLang and Azure AI Foundry all serve this OpenAI-compatible API.

Settings come from the environment, or else from .env in the repository root:
LLM_BASE_URL and LLM_API_KEY  the server of the agents, by default Ollama on this machine
EMBED_BASE_URL, EMBED_API_KEY the server of the embedding model, by default the same as the agents'
LLM_MODEL, EMBED_MODEL        one model for every agent and the embedding model, read by graph.py
"""

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Values already in the environment win over the file. The file may also set SSL_CERT_FILE for a network that
# re-signs TLS, which the HTTP client reads when it is created below.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# A hosted endpoint can answer 429 when its rate limit is reached; the client waits and retries.
client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LLM_API_KEY", "none"), timeout=1800, max_retries=6)
embed_client = client if not os.getenv("EMBED_BASE_URL") else OpenAI(
    base_url=os.getenv("EMBED_BASE_URL"), api_key=os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", "none")),
    timeout=600, max_retries=6)

MAX_TOKENS = 8192
# OpenAI's reasoning models count their reasoning within max_completion_tokens, so they get more room.
MAX_COMPLETION_TOKENS = 32768


def openai_reasoning(model):
    """OpenAI's reasoning models, gpt-5 and the o series, which take max_completion_tokens and no temperature."""
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def user(text, images=()):
    """A user message. Images travel inline as base64 data URLs."""
    if not images:
        return {"role": "user", "content": text}
    parts = [{"type": "text", "text": text}]
    for path in images:
        kind = "png" if Path(path).suffix == ".png" else "jpeg"
        data = base64.b64encode(Path(path).read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}})
    return {"role": "user", "content": parts}


def chat(model, messages, tools=None, effort=None):
    """One completion. Returns the reply as a plain message dict and a trace record.

    effort is the reasoning effort of a thinking model. "none" turns thinking off where the
    model allows it, and "low" shortens it.
    """
    start = time.perf_counter()
    options = {"tools": tools} if tools else {}
    if effort:
        options["reasoning_effort"] = effort
    if openai_reasoning(model):
        # These models reject max_tokens, and reject temperature unless reasoning is off, so they sample at their default.
        options["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        if tools and effort:
            # From gpt-5.4 on, chat completions reject tools with any effort but "none" (reasoning with tools needs the
            # Responses API). The engineer only copies values into tool calls, which needs no reasoning.
            options["reasoning_effort"] = "none"
    else:
        options |= {"temperature": 0, "max_tokens": MAX_TOKENS}
    r = client.chat.completions.create(model=model, messages=messages, **options)
    reply = r.choices[0].message.model_dump(exclude_none=True)
    record = {"model": model, "seconds": round(time.perf_counter() - start, 1),
              "tokens_in": r.usage.prompt_tokens, "tokens_out": r.usage.completion_tokens,
              "finish": r.choices[0].finish_reason, "reply": reply}
    return reply, record


def parse(content):
    """The first JSON object in a reply. Text around it is ignored, such as a code fence or a brace in K_{R}."""
    for start in (i for i, c in enumerate(content) if c == "{"):
        try:
            return json.JSONDecoder().raw_decode(content[start:])[0]
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in the reply: {content[:200]!r}")


def ask_json(model, system, text, images=(), effort=None):
    """Ask for one JSON object. A thinking model returns its reasoning in a separate field."""
    reply, record = chat(model, [{"role": "system", "content": system}, user(text, images)], effort=effort)
    if not reply.get("content"):  # a thinking model can spend the whole token budget on reasoning
        raise ValueError(f"{model} returned no answer (finish: {record['finish']}, {record['tokens_out']} tokens)")
    return parse(reply["content"]), record


def embed(model, texts):
    """Unit-norm embeddings, so a dot product is a cosine similarity."""
    # A hosted embedding model rejects an empty input, which a corpus can hold.
    texts = [t if t.strip() else "(empty)" for t in texts]
    rows = []
    for i in range(0, len(texts), 64):
        rows += [d.embedding for d in embed_client.embeddings.create(model=model, input=texts[i:i + 64]).data]
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
