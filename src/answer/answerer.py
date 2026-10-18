from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3ForCausalLM,
)
from ..data_models import (MinimalSource,
                           StudentSearchResults,
                           StudentSearchResultsAndAnswer,
                           MinimalAnswer)
import torch
from pathlib import Path
import json
from tqdm import tqdm

class Answerer():
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer: PreTrainedTokenizerBase =\
            AutoTokenizer.from_pretrained(model_name)
        self.model: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def _make_prompt(self, question: str, sources: list[MinimalSource]) -> str:
        context = ""
        for source in sources:
            context += f"{source.text}\n\n"
        return (
        "You are an assistant that answers questions based solely "
        "on the context provided below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
        )

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        prompt = self._make_prompt(question, sources)
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        if not isinstance(text, str):
            raise TypeError(
                "apply_chat_template should return a str with tokenize=False"
            )
        inputs = self.tokenizer(
            text, return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        outputs = self.model.generate( # type: ignore[attr-defined]
        **inputs,
        max_new_tokens=255,
        do_sample=False,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        decoded = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        if not isinstance(decoded, str):
            raise TypeError("decode should return a str for a single sequence")
        return decoded

    def answer_dataset(
        self,
        student_search_result_path: str,
        save_path: Path
    ):
        answerer = Answerer()
        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_result_path).read_text()
        )
        
        answers_result: list[MinimalAnswer] = []
        for search_result in tqdm(student_results.search_results, desc="answering"):
            answer = answerer.answer(
                search_result.question,
                search_result.retrieved_sources
            )
            answers_result.append(
                MinimalAnswer(
                    **search_result.model_dump(),
                    answer=answer
                )
            )

        search_result_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers_result,
            k=student_results.k
        )
        with open(save_path, "w") as file:
            json.dump(search_result_and_answer.model_dump(), file, indent=2)
