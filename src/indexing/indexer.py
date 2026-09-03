from .chunker import Chunker
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from tqdm import tqdm
from ..utils import Utils

class Indexer(Utils):
    def __init__(self) -> None:
        self.chunker: Chunker = Chunker()
        self.bm25: BM25Okapi | None = None

    def index(self):
        self.chunker.chunck_all_files()
        if not self.chunker.sources:
            raise ValueError("No sources found to index.")
        tokenized_corpus = [self.tokenize(source.text) for source
                            in tqdm(self.chunker.sources, desc="Tokenizing")]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        persisted_data = {
            "bm25": self.bm25,
            "chunks": self.chunker.sources,
        }
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "index.pkl", "wb") as file:
            pickle.dump(persisted_data, file)
