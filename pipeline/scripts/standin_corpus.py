"""A stand-in corpus from the released knowledge graph, for a test run without the papers' passages and figures.

python pipeline/scripts/standin_corpus.py <out_dir>

Each paper's passages are its title followed by the distinct sentences of its triples (the "relation" field
quotes or paraphrases the paper), packed into passages of at most 2,000 characters. There are no figures, and
few sentences give a number with its unit, so a designer finds less to cite than in the papers themselves.
kg.jsonl is written from the same triples, as kg_from_release.py does.
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

repo, out = Path(__file__).resolve().parents[2], Path(sys.argv[1])
graph = repo / "knowledge_graph/graph"
papers = {r["paper_id"]: r for r in csv.DictReader(open(graph / "papers.csv", newline="", encoding="utf-8"))}
sentences = defaultdict(dict)
out.mkdir(parents=True, exist_ok=True)
with gzip.open(graph / "triples.csv.gz", "rt", newline="", encoding="utf-8") as f, \
        open(out / "kg.jsonl", "w", encoding="utf-8") as kg:
    for row in csv.DictReader(f):
        doi = papers.get(row["paper_id"], {}).get("doi") or row["paper_id"].replace("_", "/", 1)
        kg.write(json.dumps({"head": row["head"], "relation": row["relation"], "tail": row["tail"], "source": doi},
                            ensure_ascii=False) + "\n")
        sentences[row["paper_id"]][row["relation"].strip()] = None

n = 0
with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
    for pid, sents in sentences.items():
        doi = papers.get(pid, {}).get("doi") or pid.replace("_", "/", 1)
        title = papers.get(pid, {}).get("title", "")
        chunk, k = title + ".", 0
        for s in list(sents) + [None]:
            if s is None or len(chunk) + 1 + len(s) > 2000:
                f.write(json.dumps({"id": f"{pid}#kg{k}", "source": doi, "text": chunk}, ensure_ascii=False) + "\n")
                n, k, chunk = n + 1, k + 1, title + "."
            if s is not None:
                chunk += " " + s
print(f"{n} passages from {len(sentences)} papers in {out}")
