from pathlib import Path
from rank_bm25 import BM25Okapi
import pickle
from ..utils import Utils
from ..data_models import MinimalSource
import numpy as np

class Retriever(Utils):
    def __init__(self) -> None:
        index_path = Path("data/processed/index.pkl")
        if not index_path.is_file:
            raise FileNotFoundError("The index wasn't created")
        with open("data/processed/index.pkl", "rb") as file:
            data = pickle.load(file)
            self.bm25: BM25Okapi = data["bm25"]
            self.chunks: list[MinimalSource] = data["chunks"]

    def retrieve_best_chunk(self, question: str, k: int) -> list[MinimalSource]:
        if k < 1:
            raise ValueError("k value cannot be inferior to 1")
        tokenized_question = self.tokenize(question)
        scores = self.bm25.get_scores(tokenized_question)
        top_score_index = np.argsort(scores)[::-1][:k]

        return [self.chunks[index] for index in top_score_index]
        