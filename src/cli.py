from .indexing.indexer import Indexer
from .retrieval.retriever import Retriever
from .answer.answerer import Answerer
from .evaluation.evaluator import Evaluator
from pathlib import Path

class Cli():
    def index(self, max_chunk_size: int = 2000):
        try:
            indexer = Indexer(max_chunk_size)
            indexer.index()
        except Exception as e:
            print(f"cannot index: {e}")
            return
        print("data indexed in data/processed/index.pkl")

    def search(self, query: str, k: int = 10):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"cannot answer: k must be a positive integer, got {k!r}")
            return
        try:
            retriever = Retriever()
            result = retriever.retrieve_best_chunk(query, k)
        except Exception as e:
            print(f"cannot search sources: {e}")
            return
        print("those source where found: ")
        for r in result:
            print(f"path: {r.file_path}")
            print(f"first char index: {r.first_character_index}")
            print(f"last char index: {r.last_character_index}")
            print(f"content: {r.text}\n")

    def search_dataset(
            self,
            dataset_path: str =\
                "datasets_public/public/UnansweredQuestions"
                "/dataset_docs_public.json",
            k: int = 10,
            save_directory: str = "data/output/search_results"):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"cannot answer: k must be a positive integer, got {k!r}")
            return
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path = Path(dataset_path)

            retriever = Retriever()
            output_path = output_dir / "student_search_results.json"
            retriever.search_dataset(input_path, k, output_path)
            print(f"search result saved in {output_path}")
        except Exception as e:
            print(f"cannot search dataset: {e}")

    def answer(self, query: str, k: int = 10):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"cannot answer: k must be a positive integer, got {k!r}")
            return
        try:
            retriever = Retriever()
            answerer = Answerer()
            sources = retriever.retrieve_best_chunk(query, k)
            print(answerer.answer(query, sources))
        except Exception as e:
            print(f"cannot answer: {e}")
            return

    def answer_dataset(
        self,
        student_search_result_path: str =\
            "data/output/search_results/student_search_results.json",
        save_directory: str = "data/output/answer_results"
    ):
        try:
            answerer = Answerer()
            save_directory_path = Path(save_directory)
            save_directory_path.mkdir(parents=True, exist_ok=True)
            save_file_path = (
                save_directory_path / "student_search_results_and_answer.json"
            )

            answerer.answer_dataset(
                student_search_result_path,
                save_file_path)
            print(f"answers saved in {save_file_path}")
        except Exception as e:
            print(f"cannot answer dataset: {e}")

    def evaluate(
        self,
        student_search_results_path: str =\
            "data/output/search_results/student_search_results.json",
        dataset_path: str =\
            "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
    ):
        evaluator = Evaluator()
        evaluator.evaluate(student_search_results_path, dataset_path)