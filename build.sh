#!/bin/bash
set -e
# Build offline artifacts (chunking + embedding + index)
PYTHONPATH=. python pipeline/build.py --max-docs 500
echo "Artifacts built."