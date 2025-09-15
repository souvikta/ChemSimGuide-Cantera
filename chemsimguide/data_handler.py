# chemsimguide/data_handler.py
"""
Utilities for loading plain-text documentation and converting it into
overlapping chunks suitable for embedding and retrieval.

Functions
---------
load_and_chunk_data(path: str, chunk_size: int = 2_000, overlap: int = 200)
    Read every *.txt* file in *path*, slice each file into overlapping
    character windows, and return a list of dicts
    ``{"text": str, "source": str}``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

def load_and_chunk_data(
    data_dir: str | Path,
    chunk_size: int = 2_000,
    overlap: int = 200,
) -> List[Dict[str, str]]:
    """
    Load all *.txt* files from *data_dir* and break each file into
    overlapping text chunks.

    Parameters
    ----------
    data_dir : str | pathlib.Path
        Directory containing the Cantera (or other) documentation in
        plain-text files.
    chunk_size : int, default 2 000
        Target number of characters per chunk.
    overlap : int, default 200
        Characters of overlap between consecutive chunks.
        A small overlap preserves sentence continuity for embedding.

    Returns
    -------
    List[Dict[str, str]]
        Each dict has keys:

        * ``text``   – the chunk content
        * ``source`` – original filename (for traceability)

        An *empty list* is returned if the directory is missing or no
        text files are found.

    Notes
    -----
    *Uses UTF-8 decoding*; files with a different encoding will raise
    `UnicodeDecodeError`.
    """

    data_dir = Path(data_dir)
    chunks: List[Dict[str, str]] = []

    if not data_dir.is_dir():
        print(f"[load_and_chunk_data] directory not found: {data_dir}")
        return chunks

    txt_files = sorted(p for p in data_dir.glob("*.txt") if p.is_file())
    if not txt_files:
        print(f"[load_and_chunk_data] no .txt files in {data_dir}")
        return chunks

    print(f"[load_and_chunk_data] processing {len(txt_files)} files...")

    for txt_path in txt_files:
        try:
            content = txt_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"  × could not read {txt_path.name}: {exc}")
            continue

        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            chunks.append({"text": chunk_text, "source": txt_path.name})

            # advance with overlap
            start = max(start + chunk_size - overlap, start + int(chunk_size * 0.1))

    print(f"[load_and_chunk_data] produced {len(chunks)} chunks.")
    return chunks
