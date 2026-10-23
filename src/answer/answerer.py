"""Generate answers with Qwen3-0.6B from retrieved context."""

import json
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3ForCausalLM,
)

from ..data_models import (MinimalAnswer,
                           MinimalSource,
                           StudentSearchResults,
                           StudentSearchResultsAndAnswer)

MODEL_NAME: str = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS: int = 255


class _TextGenerator(Protocol):
    device: torch.device

    def generate(self, **kwargs: Any) -> torch.Tensor:
        ...


class Answerer():
    """Answer questions with Qwen3-0.6B, using retrieved chunks as context."""

    def __init__(self) -> None:
        """Load the tokenizer and the model.

        The model runs on the GPU in float16 if CUDA is available, otherwise
        on the CPU in float32.
        """
        use_cuda = torch.cuda.is_available()
        self.tokenizer: PreTrainedTokenizerBase = \
            AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model: _TextGenerator = cast(
            _TextGenerator,
            Qwen3ForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if use_cuda else torch.float32,
                device_map="auto" if use_cuda else None,
            ),
        )

    def _make_prompt(self, question: str, sources: list[MinimalSource]) -> str:
        """Build the prompt from the question and the retrieved chunks.

        Args:
            question: The question to answer.
            sources: Chunks used as context.

        Returns:
            The prompt text, before the chat template is applied.
        """
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
        """Generate an answer to a question from the given chunks.

        Generation is greedy (deterministic) and the thinking mode of Qwen3
        is disabled.

        Args:
            question: The question to answer.
            sources: Chunks used as context.

        Returns:
            The generated answer, without the prompt.

        Raises:
            TypeError: If the tokenizer returns an unexpected type.
        """
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

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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
            save_path: Path) -> None:
        """Answer every question of a search result file.

        Args:
            student_search_result_path: Path of a StudentSearchResults JSON
                file.
            save_path: Path of the output file
                (StudentSearchResultsAndAnswer JSON).
        """
        student_results = StudentSearchResults.model_validate_json(
            Path(student_search_result_path).read_text()
        )

        answers_result: list[MinimalAnswer] = []
        for search_result in tqdm(student_results.search_results,
                                  desc="answering"):
            answer = self.answer(
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
