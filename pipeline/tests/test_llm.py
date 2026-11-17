# The request options chat sends to each kind of model, with no server. python -m pipeline.tests.test_llm
from types import SimpleNamespace

from pipeline import llm

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

sent = []
def create(**kwargs):
    sent.append(kwargs)
    message = SimpleNamespace(model_dump=lambda exclude_none: {"role": "assistant", "content": "{}"})
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

llm.chat("gpt-oss:20b", [], effort="medium")
check("a local model gets temperature 0 and max_tokens", sent[-1].get("temperature") == 0 and "max_tokens" in sent[-1]
      and sent[-1]["reasoning_effort"] == "medium", sent[-1])
llm.chat("gpt-oss:20b", [], TOOLS, effort="low")
check("a local model keeps its effort with tools", sent[-1]["reasoning_effort"] == "low", sent[-1])
llm.chat("gpt-5.4", [], effort="medium")
check("gpt-5.4 gets max_completion_tokens and no temperature",
      "temperature" not in sent[-1] and "max_tokens" not in sent[-1] and "max_completion_tokens" in sent[-1], sent[-1])
check("gpt-5.4 keeps its effort without tools", sent[-1]["reasoning_effort"] == "medium", sent[-1])
llm.chat("gpt-5.4", [], TOOLS, effort="low")
check("gpt-5.4 with tools runs without reasoning", sent[-1]["reasoning_effort"] == "none", sent[-1])
llm.chat("gpt-5.4", [], effort="none")
check("gpt-5.4 with effort none keeps it", sent[-1]["reasoning_effort"] == "none", sent[-1])
llm.chat("gpt-5.4", [])
check("no effort sends no reasoning_effort", "reasoning_effort" not in sent[-1], sent[-1])

texts = []
def embeddings(model, input):
    texts.extend(input)
    return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])
llm.embed_client = SimpleNamespace(embeddings=SimpleNamespace(create=embeddings))
v = llm.embed("text-embedding-3-large", ["a", "", "  "] + ["b"] * 70)
check("embed sends no empty text and returns one unit vector per text",
      "" not in texts and "  " not in texts and v.shape == (73, 2) and abs(v[0, 0] - 1) < 1e-6, (texts[:3], v.shape))

print("FAILED" if bad else "ALL PASS", bad)
