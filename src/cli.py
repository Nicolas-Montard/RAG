from .indexing.indexer import Indexer
from .retrieval.retriever import Retriever
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
            self, dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results/"):
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(dataset_path)
        try:
            retriever = Retriever()
            retriever.search_dataset(input_path, k, output_dir)
            print(f"search result saved in {output_dir}")
        except Exception as e:
            print(f"cannot search dataset: {e}")