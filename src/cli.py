from .indexing.indexer import Indexer
from .retrieval.retriever import Retriever

class Cli():
    def index(self, max_chunk_size: int = 2000):
        try:
            indexer = Indexer(max_chunk_size)
            indexer.index()
        except Exception as e:
            print(f"cannot index: {e}")
            return
        print("data indexed in data/precessed/index.pkl")

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