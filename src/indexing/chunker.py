"""Split the files of the vLLM repository into chunks for indexing."""

from pathlib import Path
from typing import Any

from langchain_text_splitters import (ExperimentalMarkdownSyntaxTextSplitter,
                                      Language,
                                      RecursiveCharacterTextSplitter)
from tqdm import tqdm

from ..data_models import MinimalSource

REPO_ROOT: str = "data/raw/vllm-0.10.1"


class Chunker():
    """Cut the repository files into MinimalSource chunks.

    Python files, markdown files and other text files each use a splitter
    adapted to their format. The chunks are stored in `sources`.
    """

    def __init__(self, max_chunck_size: int) -> None:
        """Create the splitters.

        Args:
            max_chunck_size: Maximum size of a chunk, in characters.
        """
        self.chunck_overlap = 100
        if max_chunck_size < 500:
            self.chunck_overlap = 0
        if max_chunck_size > 1750:
            max_chunck_size = 1750
        self.chunck_size = max_chunck_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunck_size,
            chunk_overlap=self.chunck_overlap,
            add_start_index=True,
            strip_whitespace=False,
        )
        self.python_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=self.chunck_size,
            chunk_overlap=self.chunck_overlap,
            add_start_index=True,
            language=Language.PYTHON,
            strip_whitespace=False,
        )
        headers_to_split_on = [
            ('#', 'headers 1'),
            ('##', 'headers 2'),
            ('###', 'headers 3')
        ]
        self.markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        self.sources: list[MinimalSource] = []

    def chunck_one_file(self, file_name: Path) -> list[dict[str, Any]]:
        """Chunk a generic text file with the default splitter.

        Args:
            file_name: Path of the file to chunk.

        Returns:
            One dict per chunk (path, character indexes and text). The list
            is empty if the file cannot be read as UTF-8.
        """
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
        except (OSError, UnicodeDecodeError):
            return []
        docs = self.text_splitter.create_documents([text])
        result: list[dict[str, Any]] = []
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            source: dict[str, Any] = {}
            source["file_path"] = str(file_name)
            source["first_character_index"] = start
            source["last_character_index"] = end
            source["text"] = doc.page_content
            result.append(source)
        return result

    def chunck_python(self, file_name: Path) -> list[dict[str, Any]]:
        """Chunk a Python file, cutting on classes and functions first.

        Args:
            file_name: Path of the Python file to chunk.

        Returns:
            One dict per chunk (path, character indexes and text). The list
            is empty if the file cannot be read as UTF-8.
        """
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
        except (OSError, UnicodeDecodeError):
            return []
        docs = self.python_splitter.create_documents([text])
        result: list[dict[str, Any]] = []
        for doc in docs:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            source: dict[str, Any] = {}
            source["file_path"] = str(file_name)
            source["first_character_index"] = start
            source["last_character_index"] = end
            source["text"] = doc.page_content
            result.append(source)
        return result

    def chunck_all_files(self) -> None:
        """Chunk every file of the repository and fill `sources`."""
        root = Path(REPO_ROOT)
        all_files = [path for path in root.rglob("*") if path.is_file()]
        for file in tqdm(all_files, desc='Chunking'):
            if file.suffix == ".py":
                chunks = self.chunck_python(file)
            elif file.suffix == ".md":
                chunks = self.chunck_markdown(file)
            else:
                chunks = self.chunck_one_file(file)
            self.fill_sources(chunks)

    def chunck_markdown(self, file_name: Path) -> list[dict[str, Any]]:
        """Chunk a markdown file, first by headers, then by size.

        The position of each header section is searched in the original
        text.

        Args:
            file_name: Path of the markdown file to chunk.

        Returns:
            One dict per chunk (path, character indexes and text). The list
            is empty if the file is empty or cannot be read as UTF-8.
        """
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
        except Exception:
            return []

        if not text.strip():
            return []

        sections = self.markdown_splitter.split_text(text)

        result: list[dict[str, Any]] = []
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
                source: dict[str, Any] = {}
                source["file_path"] = str(file_name)
                source["first_character_index"] = doc_start
                source["last_character_index"] = doc_end
                source["text"] = doc.page_content
                result.append(source)

        return result

    def fill_sources(self, chunks: list[dict[str, Any]]) -> None:
        """Convert chunk dicts to MinimalSource and store them.

        Args:
            chunks: Chunk dicts produced by one of the chunck_* methods.
        """
        for chunk in chunks:
            self.sources.append(MinimalSource(**chunk))
