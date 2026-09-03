import re

class Utils():
    @staticmethod
    def tokenize(text: str)-> list[str]:
        return re.findall(r"\w+", text.lower())