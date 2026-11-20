"""Draft eval questions from your documents with Gemini, for you to review.

    python -m eval.build_questions --n 30

Writes eval/questions.draft.jsonl. Read every item, fix or delete bad ones,
then save the final set as eval/questions.jsonl. Once a run has started, do
not change questions.jsonl (run_eval records its hash and refuses to resume
on a changed set).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from app import llm
from app.config import ConfigError, load_settings
from app.ingest import load_chunks

SYSTEM = (
    "You write evaluation questions for a document question-answering system. "
    "Given one passage, write ONE question that the passage answers on its own, plus a short reference answer "
    "listing the key facts. The question must be specific and must not mention 'the passage' or 'the document'. "
    'Respond with JSON only: {"question": "...", "reference": "..."}'
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=30, help="number of questions to draft")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="eval/questions.draft.jsonl")
    args = p.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if settings.llm_backend == "fake":
        print("error: set LLM_BACKEND=gemini_api to draft real questions", file=sys.stderr)
        return 2
    llm.configure(settings)

    chunks = [c for c in load_chunks(Path(settings.docs_dir), settings.chunk_size, settings.chunk_overlap) if len(c.text) > 200]
    if not chunks:
        print(f"error: no usable documents in {settings.docs_dir}", file=sys.stderr)
        return 1

    # Spread questions across documents: round-robin over docs, random chunk per doc.
    rng = random.Random(args.seed)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picked = []
    while len(picked) < args.n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < args.n:
                picked.append(by_doc[d].pop(rng.randrange(len(by_doc[d]))))

    out = Path(args.out)
    written = 0
    with out.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, 1):
            resp = llm.generate("question_writer", SYSTEM, [{"role": "user", "content": chunk.text}], json_output=True)
            data = llm.parse_json(resp.text) or {}
            if not data.get("question") or not data.get("reference"):
                print(f"skip {chunk.chunk_id}: unusable output", file=sys.stderr)
                continue
            f.write(json.dumps({
                "id": f"q{i:03d}",
                "question": data["question"].strip(),
                "reference": data["reference"].strip(),
                "source_doc_id": chunk.doc_id,
                "source_chunk_id": chunk.chunk_id,
            }) + "\n")
            written += 1
            print(f"[{i}/{len(picked)}] {chunk.doc_id}")
    print(f"wrote {written} draft questions to {out}. Review them, then save as eval/questions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
