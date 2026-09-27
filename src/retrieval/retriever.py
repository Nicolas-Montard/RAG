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
        question_list: list[UnansweredQuestion] = []
        for question in data:
            question_list.append(UnansweredQuestion(**question))
        search_results = []
        for question in question_list:
            minSearchResult = MinimalSearchResults(
                **question.model_dump(),
                retrieved_sources=self.retrieve_best_chunk(
                    question.question, k))
            search_results.append(minSearchResult)
        with open(save_dir, "w") as file:
            json.dump(search_results, file)