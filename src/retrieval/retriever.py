"""Retrieve the most relevant chunks of the index for a query."""

import json
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..data_models import (MinimalSearchResults, MinimalSource, RagDataset,
                           StudentSearchResults)
from ..indexing.indexer import INDEX_DIR, INDEX_FILE
from ..utils import Utils


class Retriever(Utils):
    """Search the BM25 index built by the Indexer."""

    def __init__(self) -> None:
        """Load the pickled index.

        Raises:
            FileNotFoundError: If the index was not created yet.
        """
        index_path = Path(INDEX_DIR) / INDEX_FILE
        if not index_path.is_file():
            raise FileNotFoundError("The index wasn't created")
        with open(index_path, "rb") as file:
            data = pickle.load(file)
        self.bm25: BM25Okapi = data["bm25"]
        self.chunks: list[MinimalSource] = data["chunks"]

    def retrieve_best_chunk(
            self, question: str, k: int) -> list[MinimalSource]:
        """Return the k chunks with the highest BM25 score.

        Args:
            question: The query, in natural language. Must not be empty
                or contain only whitespace.
            k: Number of chunks to return. Must be at least 1.

        Returns:
            The chunks, best score first.

        Raises:
            ValueError: If the query is empty or k is lower than 1.
        """
        if not question.strip():
            raise ValueError("query cannot be empty")
        if k < 1:
            raise ValueError("k value cannot be inferior to 1")
        tokenized_question = self.tokenize(question)
        scores = self.bm25.get_scores(tokenized_question)
        top_score_index = np.argsort(scores)[::-1][:k]

        return [self.chunks[index] for index in top_score_index]

    def search_dataset(
            self, data_path: Path, k: int, save_dir: Path) -> None:
        """Retrieve the best chunks for every question of a dataset.

        Args:
            data_path: JSON file of questions (RagDataset format).
            k: Number of chunks to retrieve per question.
            save_dir: Path of the output file (StudentSearchResults JSON).
        """
        dataset = RagDataset.model_validate_json(Path(data_path).read_text())
        output_path = Path(
            f"{save_dir}/{data_path.stem}_search{data_path.suffix}")

        search_results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="search data"):
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=self.retrieve_best_chunk(
                        question.question, k),
                )
            )

        student_search_results = StudentSearchResults(
            search_results=search_results,
            k=k,
        )

        with open(output_path, "w") as file:
            json.dump(student_search_results.model_dump(), file, indent=2)
