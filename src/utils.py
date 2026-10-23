"""Shared helpers for the indexing and retrieval steps."""

import re


class Utils():
    """Text helpers shared by the Indexer and the Retriever."""

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split a text into lowercase tokens for BM25.

        Each identifier (letters, digits and underscores) is kept whole,
        and also split on snake_case and camelCase boundaries. For example
        "PagedAttention" gives "pagedattention", "paged" and "attention".

        The same function must be used to index the chunks and to tokenize
        the queries, so the index must be rebuilt after any change.

        Args:
            text: The text to tokenize.

        Returns:
            The list of tokens, in order of appearance.
        """
        tokens: list[str] = []
        for word in re.findall(r"[A-Za-z0-9_]+", text):
            tokens.append(word.lower())
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
            parts = spaced.replace("_", " ").lower().split()
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens
