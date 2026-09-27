from langchain_text_splitters import (RecursiveCharacterTextSplitter,
                                      Language,
                                      ExperimentalMarkdownSyntaxTextSplitter)

from ..data_models import MinimalSource
from pathlib import Path
from tqdm import tqdm

class Chunker():
    def __init__(self, max_chunck_size) -> None:
        self.chunck_size = max_chunck_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunck_size,
            chunk_overlap=200,
            add_start_index=True
        )
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=self.chunck_size,
            chunk_overlap=200,
            add_start_index=True,
            language=Language.PYTHON
        )
        headers_to_split_on = [
            ('#', 'headers 1'),
            ('##', 'headers 2'),
            ('###', 'headers 3')
        ]
        self.markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        self.sources: list[MinimalSource] = []

    def chunck_one_file(self, file_name: Path) -> list[dict]:
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
        except:
            return []
        docs = self.text_splitter.create_documents([text])
        result = []
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            source = {}
            source["file_path"] = str(file_name)
            source["first_character_index"] = start
            source["last_character_index"] = end
            source["text"] = doc.page_content
            result.append(source)
        return result

    def chunck_python(self, file_name: Path) -> list[dict]:
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
        except:
            return []
        docs = self.python_splitter.create_documents([text])
        result = []
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            source = {}
            source["file_path"] = str(file_name)
            source["first_character_index"] = start
            source["last_character_index"] = end
            source["text"] = doc.page_content
            result.append(source)
        return result

    def chunck_all_files(self):
        root = Path("data/raw/vllm-0.10.1")
        all_files = [path for path in root.rglob("*") if path.is_file()]
        for file in tqdm(all_files, desc='Chunking'):
            if file.suffix == ".py":
                chunks = self.chunck_python(file)
                self.fill_sources(chunks)
                pass
            elif file.suffix == ".md":
                chunks = self.chunck_markdown(file)
                self.fill_sources(chunks)
            else:
                chunks = self.chunck_one_file(file)
                self.fill_sources(chunks)
                pass

    def chunck_markdown(self, file_name: Path) -> list[dict]:
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                    text = file.read()
        except:
            return []

        if not text.strip():
            return []

        sections = self.markdown_splitter.split_text(text)

        result: list[dict] = []
        cursor = 0
        for section in sections:
            docs = self.text_splitter.create_documents([section.page_content])

            section_start = text.find(section.page_content, cursor)
            if section_start == -1:
                continue
            cursor = section_start + len(section.page_content)

            for doc in docs:
                doc_start = doc.metadata["start_index"] + section_start
                doc_end = doc_start + len(doc.page_content)
                source = {}
                source["file_path"] = str(file_name)
                source["first_character_index"] = doc_start
                source["last_character_index"] = doc_end
                source["text"] = doc.page_content
                result.append(source)

        return result

    def fill_sources(self, chunks: list[dict]):
        for chunk in chunks:
            self.sources.append(MinimalSource(**chunk))

