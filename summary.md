# Project Folder and File Structure

Here's what every folder and file is for, in simple terms.

## Folders

| Folder         | What it's for                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `engine/`      | The heart of the project — the search algorithms (HNSW graph search, BM25 keyword search, mixing them together). **Pure Python, not built yet.**       |
| `pipeline/`    | The offline workshop — turns raw text into searchable data (chunking, turning text into numbers, saving files).                                        |
| `experiments/` | The A/B testing brain — decides which search setting a user gets, logs results, and runs the math to tell if one setting is better. **Not built yet.** |
| `api/`         | The web server — the part users and the website talk to. **Just a stub so far.**                                                                       |
| `web/`         | The website (React) — the search box and results page. **Empty, later.**                                                                               |
| `bench/`       | Speed tests — scripts that measure how fast and accurate our search is. **Empty, later.**                                                              |
| `eval/`        | The answer key — a list of questions with correct answers, so we can score how good the search is. **Empty, later.**                                   |
| `docs/`        | Notes — design decisions and explanations. **Empty, later.**                                                                                           |
| `data/`        | The built product — the chopped text and its number representations. Created by us, **not in git**.                                                    |
| `tests/`       | Checks that our code works correctly.                                                                                                                  |
| `.venv/`       | The project's private Python environment — its own installed packages, separate from the rest of your computer.                                        |

## Files

| File               | What it does                                                       |
| ------------------ | ------------------------------------------------------------------ |
| `Plan.md`          | The full blueprint — goals, architecture, and 2-week schedule.     |
| `README.md`        | The intro page — what the project is and how to get started.       |
| `requirements.txt` | Shopping list of Python packages to install.                       |
| `.gitignore`       | Tells Git which files not to track (the venv, data files, caches). |

## Inside `pipeline/`

This is the **only part with real code** right now.

| File            | What it does                                                                                    |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `chunking.py`   | Splits long articles into small overlapping pieces of ~200 words.                               |
| `embeddings.py` | Turns text pieces into 384 numbers using the MiniLM model, plus a cosine similarity helper.     |
| `build.py`      | The main script — downloads the dataset, chunks it, embeds it, and saves everything to `data/`. |

## Inside `api/`

| File      | What it does                                                            |
| --------- | ----------------------------------------------------------------------- |
| `main.py` | A tiny FastAPI server with one route (`/health`) so we know it's alive. |

## Inside `tests/`

| File               | What it does                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| `test_chunking.py` | Verifies that the chunker splits text correctly (right sizes, right overlap, and handles short text). |

## Simple Analogy

Think of the project like a restaurant:

* 🍳 **`pipeline/`** → The kitchen preparing the ingredients
* 🔎 **`engine/`** → The cooks searching/finding the right ingredients
* 🧑‍🍳 **`api/`** → The waiter connecting customers to the kitchen
* 🌐 **`web/`** → The menu customers see
* 🧊 **`data/`** → The prepared food stored in the fridge
* 🧪 **`tests/`** → Quality checks to make sure everything works
* 📊 **`experiments/`** → Testing which recipe works better
* 📚 **`docs/`** → The restaurant's notes and instructions
