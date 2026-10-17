#!/bin/bash
set -e
# Build offline artifacts (chunking + embedding + index).
# Full WikiText-2 train split: ~14.3k paragraphs -> ~16.8k chunks over ~600 articles.
PYTHONPATH=. python pipeline/build.py --max-docs 0
echo "Artifacts built."
