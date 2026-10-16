from pathlib import Path

from pipeline.chunking import chunk_text


def test_chunk_text_basic():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text("d0", text, size=200, overlap=40)
    assert chunks[0].text.split()[0] == "w0"
    assert chunks[1].start_word == 160  # 200 - 40 overlap
    # every chunk id unique and prefixed
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("d0::") for i in ids)


def test_chunk_text_short():
    chunks = chunk_text("d0", "hello world", size=200, overlap=40)
    assert len(chunks) == 1
    assert chunk_text("d0", "   ", size=200, overlap=40) == []


def test_chunk_text_invalid():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("d0", "a b c", size=10, overlap=10)
