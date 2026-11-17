"""Write the corpus's kg.jsonl from the released triples in knowledge_graph/graph/.

python pipeline/scripts/kg_from_release.py <corpus>/ [knowledge_graph/graph/triples.csv.gz]

triples.csv.gz holds one triple per row with head, relation, tail, paper_id and corpus. Each becomes
{"head", "relation", "tail", "source"} with the DOI of its paper as source, as build_corpus.py writes
from per-paper triple files. An existing kg.jsonl is replaced.
"""

import csv
import gzip
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
release = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parents[2] / "knowledge_graph/graph/triples.csv.gz")
dois = {}
with open(release.with_name("papers.csv"), newline="", encoding="utf-8") as f:
    dois = {row["paper_id"]: row["doi"] for row in csv.DictReader(f)}

out.mkdir(parents=True, exist_ok=True)
n = 0
with gzip.open(release, "rt", newline="", encoding="utf-8") as f, open(out / "kg.jsonl", "w", encoding="utf-8") as kg:
    for row in csv.DictReader(f):
        source = dois.get(row["paper_id"]) or row["paper_id"].replace("_", "/", 1)
        kg.write(json.dumps({"head": row["head"], "relation": row["relation"], "tail": row["tail"], "source": source},
                            ensure_ascii=False) + "\n")
        n += 1
print(f"{n} triples from {len(dois)} papers written to {out / 'kg.jsonl'}")
