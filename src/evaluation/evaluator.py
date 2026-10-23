"""Compute the recall@k of search results against a ground truth."""

import json
from pathlib import Path
from typing import Any

from ..data_models import MinimalSource, StudentSearchResults

DEFAULT_SEARCH_RESULTS_PATH: str = (
    "data/output/search_results/student_search_results.json"
)
DEFAULT_DATASET_PATH: str = (
    "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
)
OVERLAP_THRESHOLD: float = 0.05


class Evaluator():
    """Measure how often the expected sources are found in the top k."""

    def evaluate(
            self,
            student_search_results_path: str = DEFAULT_SEARCH_RESULTS_PATH,
            dataset_path: str = DEFAULT_DATASET_PATH) -> None:
        """Print recall@1, @3, @5 and @10.

        This is for personal testing only: the official score is computed
        by the moulinette. Errors are printed, not raised.

        Args:
            student_search_results_path: Path of a StudentSearchResults JSON
                file, produced by search_dataset.
            dataset_path: Path of the AnsweredQuestions JSON file.
        """
        try:
            student_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text()
            )
            dataset: list[dict[str, Any]] = \
                json.loads(Path(dataset_path).read_text())["rag_questions"]

            k_arrays = self.get_k_array(student_results, dataset)
            if not k_arrays["k_array_1"]:
                print("cannot evaluate: no question to evaluate")
                return
            recall_1 = sum(k_arrays["k_array_1"]) / len(k_arrays["k_array_1"])
            recall_3 = sum(k_arrays["k_array_3"]) / len(k_arrays["k_array_3"])
            recall_5 = sum(k_arrays["k_array_5"]) / len(k_arrays["k_array_5"])
            recall_10 = \
                sum(k_arrays["k_array_10"]) / len(k_arrays["k_array_10"])

            print("Total number of question with sources: "
                  f"{len(dataset)}")
            print("Total number of questions with student sources: "
                  f"{len(student_results.search_results)}\n")
            print("Evaluation results:\n")
            print(f"Recall@1: {recall_1:.3f} ({recall_1 * 100:.1f}%)")
            print(f"Recall@3: {recall_3:.3f} ({recall_3 * 100:.1f}%)")
            print(f"Recall@5: {recall_5:.3f} ({recall_5 * 100:.1f}%)")
            print(f"Recall@10: {recall_10:.3f} ({recall_10 * 100:.1f}%)")
        except Exception as e:
            print(f"cannot evaluate: {e}")

    def get_source_validity(
            self,
            stud_source: MinimalSource,
            valid_data: dict[str, Any]) -> float:
        """Return how much of an expected source a retrieved chunk covers.

        Args:
            stud_source: A chunk retrieved by the student's retriever.
            valid_data: A ground-truth question, with a "sources" list.

        Returns:
            The best overlap ratio, between 0 and 1, over the expected
            sources of the same file. It is 0 if no source matches.
        """
        best = 0.0
        for valid in valid_data["sources"]:
            if valid["file_path"] != stud_source.file_path:
                continue
            length = (valid["last_character_index"]
                      - valid["first_character_index"])
            if length <= 0:
                continue
            overlap = (
                min(valid["last_character_index"],
                    stud_source.last_character_index)
                - max(valid["first_character_index"],
                      stud_source.first_character_index)
            )
            best = max(best, max(0, overlap) / length)
        return best

    def get_k_array(
            self,
            student_results: StudentSearchResults,
            dataset: list[dict[str, Any]]) -> dict[str, list[int]]:
        """Tell, for each ground-truth question, if it is found in the top k.

        A question counts as found when at least one of the k first
        retrieved chunks overlaps an expected source by more than
        OVERLAP_THRESHOLD. A question missing from the student results
        counts as not found.

        Args:
            student_results: The student's search results.
            dataset: The ground-truth questions.

        Returns:
            For k in 1, 3, 5 and 10, a list with one 0 or 1 per
            ground-truth question, under the key "k_array_<k>".
        """
        k_array_10: list[int] = []
        k_array_5: list[int] = []
        k_array_3: list[int] = []
        k_array_1: list[int] = []

        for valid_data in dataset:
            stud_question = next(
                (stud_result for stud_result in student_results.search_results
                 if stud_result.question_id == valid_data["question_id"]),
                None)
            source_correlation: list[float] = []
            if stud_question is not None:
                source_correlation = [
                    self.get_source_validity(stud_source, valid_data)
                    for stud_source in stud_question.retrieved_sources
                ]

            k_array_1.append(int(
                max(source_correlation[:1], default=0.0)
                > OVERLAP_THRESHOLD))
            k_array_3.append(int(
                max(source_correlation[:3], default=0.0)
                > OVERLAP_THRESHOLD))
            k_array_5.append(int(
                max(source_correlation[:5], default=0.0)
                > OVERLAP_THRESHOLD))
            k_array_10.append(int(
                max(source_correlation[:10], default=0.0)
                > OVERLAP_THRESHOLD))

        return {"k_array_1": k_array_1,
                "k_array_3": k_array_3,
                "k_array_5": k_array_5,
                "k_array_10": k_array_10}
