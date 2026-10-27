"""Command line interface of the RAG pipeline."""

from pathlib import Path

from .answer.answerer import Answerer
from .evaluation.evaluator import Evaluator
from .indexing.indexer import Indexer
from .retrieval.retriever import Retriever

MaxChunkSize: int = 2000

DEFAULT_QUESTIONS_PATH: str = (
    "datasets_public/public/UnansweredQuestions/dataset_docs_public.json"
)
DEFAULT_ANSWERED_PATH: str = (
    "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
)
DEFAULT_SEARCH_RESULTS_PATH: str = (
    "data/output/search_results/dataset_docs_public_search.json"
)


class Cli():
    """Commands exposed through Fire: index, search, answer and evaluate."""

    def index(self, max_chunk_size: int = MaxChunkSize) -> None:
        """Chunk the vLLM repository and build the BM25 index.

        The index is saved in data/processed/index.pkl.

        Args:
            max_chunk_size: Maximum size of a chunk, in characters.
                Must be between 1 and MaxChunkSize.
        """
        if not isinstance(max_chunk_size, int) or \
                isinstance(max_chunk_size, bool) or \
                max_chunk_size < 1 or max_chunk_size > MaxChunkSize:
            print("cannot index: max_chunk_size must be an int superior to 0 "
                  f"and inferior or equal to {MaxChunkSize}, "
                  f"got {max_chunk_size!r}")
            return
        try:
            indexer = Indexer(max_chunk_size)
            indexer.index()
        except Exception as e:
            print(f"cannot index: {e}")
            return
        print("data indexed in data/processed/index.pkl")

    def search(self, query: str, k: int = 10) -> None:
        """Print the k best chunks for a single query.

        Args:
            query: The question to search for.
            k: Number of chunks to retrieve. Must be a positive integer.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"cannot search: k must be a positive integer, got {k!r}")
            return
        if not isinstance(query, str):
            print(f"cannot search: the query must be a str, got {query!r}")
            return
        try:
            retriever = Retriever()
            result = retriever.retrieve_best_chunk(query, k)
        except Exception as e:
            print(f"cannot search sources: {e}")
            return
        print("those sources were found:")
        for r in result:
            print(f"path: {r.file_path}")
            print(f"first char index: {r.first_character_index}")
            print(f"last char index: {r.last_character_index}")
            print(f"content: {r.text}\n")

    def search_dataset(
            self,
            dataset_path: str = DEFAULT_QUESTIONS_PATH,
            k: int = 10,
            save_directory: str = "data/output/search_results") -> None:
        """Retrieve the best chunks for every question of a dataset.

        The result is written as a StudentSearchResults JSON file named
        student_search_results.json.

        Args:
            dataset_path: Path to the JSON file of questions.
            k: Number of chunks to retrieve per question.
            save_directory: Directory where the result file is written.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print("cannot search dataset: k must be a positive integer, "
                  f"got {k!r}")
            return
        try:
            output_dir = Path(save_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_path = Path(dataset_path)

            retriever = Retriever()
            retriever.search_dataset(input_path, k, output_dir)
            print(f"search result saved in {output_dir}")
        except Exception as e:
            print(f"cannot search dataset: {e}")

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: Number of chunks given to the model as context.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"cannot answer: k must be a positive integer, got {k!r}")
            return
        if not isinstance(query, str):
            print(f"cannot answer: the query must be a str, got {query!r}")
            return
        try:
            retriever = Retriever()
            answerer = Answerer()
            sources = retriever.retrieve_best_chunk(query, k)
            print(answerer.answer(query, sources))
        except Exception as e:
            print(f"cannot answer: {e}")

    def answer_dataset(
            self,
            student_search_result_path: str = DEFAULT_SEARCH_RESULTS_PATH,
            save_directory: str = "data/output/answer_results") -> None:
        """Generate an answer for every question of a search result file.

        The result is written as a StudentSearchResultsAndAnswer JSON file
        named student_search_results_and_answer.json.

        Args:
            student_search_result_path: Path to the file produced by
                search_dataset.
            save_directory: Directory where the result file is written.
        """
        try:
            answerer = Answerer()
            save_directory_path = Path(save_directory)
            save_directory_path.mkdir(parents=True, exist_ok=True)

            answerer.answer_dataset(
                student_search_result_path,
                save_directory_path)
            print(f"answers saved in {save_directory}")
        except Exception as e:
            print(f"cannot answer dataset: {e}")

    def evaluate(
            self,
            student_search_results_path: str = DEFAULT_SEARCH_RESULTS_PATH,
            dataset_path: str = DEFAULT_ANSWERED_PATH) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        This is for personal testing only: the official score is computed
        by the moulinette.

        Args:
            student_search_results_path: Path to the file produced by
                search_dataset.
            dataset_path: Path to the AnsweredQuestions dataset.
        """
        try:
            evaluator = Evaluator()
            evaluator.evaluate(student_search_results_path, dataset_path)
        except Exception as error:
            print(f"cannot evaluate: {error}")
