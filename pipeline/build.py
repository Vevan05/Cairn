"""Offline build: load corpus → chunk → embed → save artifacts to disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.chunking import chunk_text
from pipeline.embeddings import embed_texts


def load_corpus(max_docs: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    docs: list[tuple[str, str]] = []
    for i, row in enumerate(ds):
        text = row["text"].strip()
        if len(text) < 200:  # skip headings/blank lines
            continue
        docs.append((f"doc{i}", text))
        if len(docs) >= max_docs:
            break
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--overlap", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    docs = load_corpus(args.max_docs)
    print(f"loaded {len(docs)} docs")

    chunks = []
    for doc_id, text in docs:
        chunks.extend(chunk_text(doc_id, text, size=args.chunk_size, overlap=args.overlap))
    print(f"built {len(chunks)} chunks")

    with open(args.out / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")

    vectors = embed_texts([c.text for c in chunks])
    np.save(args.out / "vectors.npy", vectors)
    print(f"saved {vectors.shape} vectors to {args.out / 'vectors.npy'}")


if __name__ == "__main__":
    main()
