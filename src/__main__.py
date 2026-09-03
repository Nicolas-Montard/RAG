from .indexing.indexer import Indexer
from .retrieval.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever()
    print(retriever.retrieve_best_chunk("how to configure an openai server", 5)[0].text)

