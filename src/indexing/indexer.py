"""Build and save the BM25 index of the vLLM repository."""

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..utils import Utils
from .chunker import Chunker

INDEX_DIR: str = "data/processed"
INDEX_FILE: str = "index.pkl"


class Indexer(Utils):
    """Chunk the repository, tokenize the chunks and save a BM25 index."""

    def __init__(self, max_chunck_size: int) -> None:
        """Create the chunker.

        Args:
            max_chunck_size: Maximum size of a chunk, in characters.
        """
        self.chunker: Chunker = Chunker(max_chunck_size)
        self.bm25: BM25Okapi | None = None

    def index(self) -> None:
        """Chunk all files, build the BM25 index and pickle it.

        The file contains a dict with the keys "bm25" (the BM25Okapi
        object) and "chunks" (the list of MinimalSource, in the same order
        as the indexed corpus).

        Raises:
            ValueError: If no chunk was produced.
        """
        self.chunker.chunck_all_files()
        if not self.chunker.sources:
            raise ValueError("No sources found to index.")
        tokenized_corpus = [
            self.tokenize(f"{source.file_path} {source.text}")
            for source in tqdm(self.chunker.sources, desc="Tokenizing")
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        persisted_data: dict[str, Any] = {
            "bm25": self.bm25,
            "chunks": self.chunker.sources,
        }
        output_dir = Path(INDEX_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / INDEX_FILE, "wb") as file:
            pickle.dump(persisted_data, file)
