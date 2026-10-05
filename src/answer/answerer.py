from transformers import AutoModelForCausalLM, AutoTokenizer
from ..data_models import MinimalSource
import torch

class Answerer():
    def __init__(self) -> None:
        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
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

    def answer(self, question: str, sources: list[MinimalSource]):
        prompt = self._make_prompt(question, sources)
        messages = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        )

        outputs = self.model.generate( # type: ignore[attr-defined]
        inputs,
        max_new_tokens=255,
        do_sample=False,
        )

        generated_tokens = outputs[0][inputs.shape[-1]:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True)