import re

class Utils():
    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        for word in re.findall(r"[A-Za-z0-9_]+", text):
            tokens.append(word.lower())
            parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
            parts = parts.replace("_", " ").lower().split()
            if len(parts) > 1:
                tokens.extend(parts)
        return tokens