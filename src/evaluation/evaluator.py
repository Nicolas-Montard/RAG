from ..data_models import StudentSearchResults, MinimalSource
from pathlib import Path
import json

class Evaluator():
    def evaluate(
            self,
            student_search_results_path: str =\
                "data/output/search_results/StudentSearchResults.json",
            dataset_path: str =\
                "datasets_public/public/AnsweredQuestions/"
                "dataset_docs_public.json",
        ):
        try:
            student_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text()
            )
            dataset: list[dict] =\
                json.loads(Path(dataset_path).read_text())["rag_questions"]

            k_arrays = self.get_k_array(student_results, dataset)
            if not k_arrays["k_array_1"]:
                print("cannot evaluate: no question to evaluate")
                return
            recall_1 = sum(k_arrays["k_array_1"]) / len(k_arrays["k_array_1"])
            recall_3 = sum(k_arrays["k_array_3"]) / len(k_arrays["k_array_3"])
            recall_5 = sum(k_arrays["k_array_5"]) / len(k_arrays["k_array_5"])
            recall_10 =\
                sum(k_arrays["k_array_10"]) / len(k_arrays["k_array_10"])

            print("Total number of question: "
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
            valid_data: dict) -> float:
        best = 0.0
        for valid in valid_data["sources"]:
            if valid["file_path"] != stud_source.file_path:
                continue
            length = valid["last_character_index"] - valid["first_character_index"]
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
            dataset: list[dict]) -> dict[str, list[int]]:
        k_array_10: list[int] = []
        k_array_5: list[int] = []
        k_array_3: list[int] = []
        k_array_1: list[int] = []

        for stud_question in student_results.search_results:
            valid_data = next(
                (data for data in dataset
                if data["question_id"] == stud_question.question_id),
                None)

            if valid_data is None:
                raise ValueError(
                    f"The question {stud_question.question_id} doesn't have "
                    "a matching question in the valid dataset")

            source_correlation: list[float] = [
                self.get_source_validity(stud_source, valid_data)
                for stud_source in stud_question.retrieved_sources
            ]

            k_array_1.append(int(max(source_correlation[:1], default=0.0)
                                 > 0.05))
            k_array_3.append(int(max(source_correlation[:3], default=0.0)
                                 > 0.05))
            k_array_5.append(int(max(source_correlation[:5], default=0.0)
                                 > 0.05))
            k_array_10.append(int(max(source_correlation[:10], default=0.0)
                                  > 0.05))

        return {"k_array_1": k_array_1,
                "k_array_3": k_array_3,
                "k_array_5": k_array_5,
                "k_array_10": k_array_10}
