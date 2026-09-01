from langchain_text_splitters import CharacterTextSplitter
from ..data_models import MinimalSource

class Chunking():
    def __init__(self) -> None:
        self.text_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.sources: list[MinimalSource] = []

    def chunck_one_file(self, file_name: str):
        self.text_splitter.split_text(file_name)