from pathlib import Path
from rank_bm25 import BM25Okapi
import pickle
from ..utils import Utils
from ..data_models import MinimalSource, UnansweredQuestion, MinimalSearchResults
import numpy as np
import json


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

    def search_dataset(self, data_path: Path, k: int, save_dir: Path):
        with open(data_path, "r") as file:
            data = json.loads(file.read())

        data = data["rag_questions"]

        question_list: list[UnansweredQuestion] = [
            UnansweredQuestion(**question) for question in data
        ]

        search_results: list[MinimalSearchResults] = []
        for question in question_list:
            min_search_result = MinimalSearchResults(
                **question.model_dump(),
                retrieved_sources=self.retrieve_best_chunk(question.question, k),
            )
            search_results.append(min_search_result)

        with open(save_dir, "w") as file:
            json.dump([r.model_dump() for r in search_results], file, indent=2)