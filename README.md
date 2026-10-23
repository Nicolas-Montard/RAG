*This project has been created as part of the 42 curriculum by nmontard.*

# RAG against the machine

A Retrieval-Augmented Generation (RAG) system that answers questions about the source code and documentation of [vLLM](https://github.com/vllm-project/vllm) (version 0.10.1).

## Description

**Goal.** Given a natural-language question about vLLM, the system finds the passages of the repository that are most relevant to it, then uses a small language model (`Qwen/Qwen3-0.6B`) to write an answer based on those passages. Every answer comes with the sources it was built from (file path and character range).

**Overview.** The pipeline has three stages:

1. **Indexing**: the repository files are cut into chunks and indexed with BM25.
2. **Retrieval**: a question is matched against the index, and the `k` best chunks are returned.
3. **Generation**: the chunks are put in a prompt and Qwen3-0.6B writes the answer.

The command line exposes each stage, and a batch mode for whole datasets of questions. Inputs and outputs are validated with pydantic models.

## Instructions

### Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- The vLLM 0.10.1 sources, placed in `data/raw/vllm-0.10.1`
- Internet access the first time, to download the Qwen3-0.6B weights from Hugging Face (they are then cached locally)

A GPU is optional. Without one, the model runs on the CPU in float32, which is much slower.

### Installation

```bash
uv sync
```

### Execution

All commands go through the package entry point:

```bash
uv run python -m src <command> [--option value]
```

| Command | Purpose |
|---|---|
| `index` | Chunk the repository and build the BM25 index |
| `search` | Print the `k` best chunks for one query |
| `search_dataset` | Retrieve chunks for every question of a dataset |
| `answer` | Answer one query using the retrieved context |
| `answer_dataset` | Answer every question of a search-results file |
| `evaluate` | Print recall@k of search results against a ground truth |

Every command takes its paths and parameters as command line arguments, and prints an error message instead of a traceback on invalid input (empty query, `k=0`, missing file, malformed JSON...).

### Makefile

A Makefile runs the commands with their default values:

| Target | Equivalent command | Purpose |
|---|---|---|
| `make install` | `uv sync` | Install the dependencies |
| `make run` | `uv run -m src index` | Build the index |
| `make search` | `uv run -m src search_dataset` | Retrieve chunks for the default dataset |
| `make answer` | `uv run -m src answer_dataset` | Answer the questions of the default search results |
| `make evaluate` | `uv run -m src evaluate` | Print recall@k with the default files |
| `make debug` | `uv run python -m pdb -m src index` | Run the indexing under the Python debugger |
| `make lint` | flake8 and mypy | Check the code style and the types |
| `make clean` | | Remove `data/output/*`, `data/processed/*` and the Python caches |

To use other values, call the commands directly: `uv run -m src <command> [--option value]`.

## Example usage

```bash
# 1. Build the index (chunks of at most 2000 characters)
uv run python -m src index --max_chunk_size 2000

# 2. Look at the 3 best chunks for one query
uv run python -m src search "How does vLLM schedule requests?" --k 3

# 3. Answer one query
uv run python -m src answer "What is PagedAttention?" --k 5

# 4. Retrieve for a whole dataset
uv run python -m src search_dataset \
    --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results

# 5. Generate the answers of the dataset
uv run python -m src answer_dataset \
    --student_search_result_path data/output/search_results/student_search_results.json \
    --save_directory data/output/answer_results

# 6. Measure recall@k against the ground truth
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/student_search_results.json \
    --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json
```

## System architecture

```
 vLLM repository                       question
        |                                 |
        v                                 v
  +-----------+   chunks   +-----------+  tokens  +-----------+
  |  Chunker  | ---------> |  Indexer  |          | Retriever |
  +-----------+            |  (BM25)   |          |  (BM25)   |
                           +-----------+          +-----------+
                                 |                      |
                            index.pkl  ---------------> |
                                                        | k best chunks
                                                        v
                                                  +-----------+
                                                  | Answerer  |
                                                  | Qwen3-0.6B|
                                                  +-----------+
                                                        |
                                                        v
                                                answer + sources
```

| Component | Module | Role |
|---|---|---|
| `Chunker` | `src/indexing/chunker.py` | Reads the files and cuts them into `MinimalSource` chunks |
| `Indexer` | `src/indexing/indexer.py` | Tokenizes the chunks, builds the BM25 index and pickles it in `data/processed/index.pkl` |
| `Retriever` | `src/retrieval/retriever.py` | Loads the index and returns the best chunks for a query |
| `Answerer` | `src/answer/answerer.py` | Builds the prompt and generates the answer with Qwen3-0.6B |
| `Evaluator` | `src/evaluation/evaluator.py` | Computes recall@1, @3, @5 and @10 |
| `Utils` | `src/utils.py` | Shared tokenizer used for both the corpus and the queries |
| `Cli` | `src/cli.py` | Command line interface, exposed with Python Fire |
| data models | `src/data_models.py` | pydantic models exchanged between the stages |

The stages communicate through files and pydantic models: `index` writes `index.pkl`; `search_dataset` writes a `StudentSearchResults` JSON; `answer_dataset` reads it and writes a `StudentSearchResultsAndAnswer` JSON.

## Chunking strategy

Chunks are produced with `langchain-text-splitters` (`RecursiveCharacterTextSplitter`), with a maximum size set by `--max_chunk_size` (default and upper limit: 2000 characters). Consecutive chunks overlap by 100 characters, except when `max_chunk_size` is below 500, where there is no overlap. The splitter depends on the file type:

- **Python files** use the Python-aware splitter, which prefers to cut on class and function boundaries before falling back to smaller separators.
- **Markdown files** are first split by headers (levels 1 to 3), then each section is cut by size.
- **Other text files** use the generic recursive splitter.

Files that cannot be read as UTF-8 (binaries, images) are skipped. Whitespace is not stripped, so that the character indexes of a chunk match the original file exactly. Each chunk is stored with its file path, its first and last character index in the original file, and its text.

Markdown sections whose text cannot be found back in the original file are skipped, and their number is reported at the end of indexing.

## Retrieval method

Retrieval uses **BM25** (`BM25Okapi` from `rank-bm25`) over the chunks. A question is tokenized, scored against every chunk, and the `k` chunks with the highest scores are returned, best first.

The tokenizer is designed for code. It extracts identifiers made of letters, digits and underscores, lowercases them, and for each identifier keeps both the whole token and its `snake_case` / `camelCase` parts. For example, `PagedAttention` gives `pagedattention`, `paged` and `attention`, so a natural-language question such as "how does paged attention work" can match code identifiers. The same function is used for the corpus and for the queries.

An empty or whitespace-only query is rejected with an error message.

## Answer generation

The `k` retrieved chunks are concatenated in a prompt that tells the model to answer only from the given context. The prompt goes through the Qwen3 chat template with thinking mode disabled, and the answer is generated greedily (`do_sample=False`, at most 255 new tokens) so that results are reproducible.

## Performance analysis

Recall@k is the share of questions for which at least one of the `k` first retrieved chunks overlaps an expected source. The official score is computed by the moulinette, and the `evaluate` command is an approximation for quick iterations (it uses a minimum overlap of 5 % of the expected source).

Final scores:

| k | Docs dataset | Code dataset |
|---|---|---|
| 1 | 0.560 | 0.404 |
| 3 | 0.740 | 0.616 |
| 5 | 0.810 | 0.657 |
| 10 | 0.850 | 0.768 |

On the code dataset (99 questions), a first measurement before the tokenizer and chunking improvements gave 0.273 / 0.384 / 0.434 / 0.495 for k = 1 / 3 / 5 / 10, so the improvements raised recall@10 from 0.495 to 0.768.

Recall grows with `k` but with diminishing returns, which means the relevant passage is often retrieved but not ranked first. Documentation questions are easier than code questions: their wording is closer to the text of the chunks, whereas code questions must be matched with identifiers. The levers used to improve the scores are the tokenizer (splitting identifiers), the chunk size and overlap, and the markdown chunking.

**Speed.** Indexing and retrieval are fast, while answer generation dominates the running time. Answering a set of about 100 questions takes on average 2.5 minutes. The time depends mostly on the prompt length (`k` and chunk size) and on the hardware: on a CPU it is much longer than on a GPU.

## Design decisions

- **BM25 instead of embeddings**: a lexical method is fast, needs no model, and works well on code where exact identifiers matter. Its limit is that it cannot match synonyms.
- **Code-aware tokenizer**: keeping the whole identifier and its parts gives exact matches and partial matches at once.
- **Character indexes stored in each chunk**: they are required by the output format and make sources verifiable. This is why whitespace is not stripped.
- **Pydantic models everywhere**: files are validated when read, so malformed input fails with a clear message.
- **Greedy decoding and thinking disabled**: reproducible answers, and no reasoning text mixed into the answers.
- **Model loaded once**: the `Answerer` loads Qwen3 in its constructor and reuses it for all questions.
- **Errors printed, not raised**: each command catches errors and prints a message, so invalid input never produces a traceback.

## Challenges faced

`enable_thinking=False` to the chat template fixed it.
- **Typing with `transformers`**: the library's type stubs make Pylance and mypy report false positives on `apply_chat_template`, `decode` and `generate`. They were handled with `isinstance` checks, and a cast where no clean option exists.
- **GPU and CPU machines**: `device_map="auto"` requires the `accelerate` package, and the inputs must be moved to the model's device with `.to(model.device)`.
- **Slow generation**: reduced by limiting the new tokens and the number of chunks given to the model, and by loading the model only once.
- **Character indexes in markdown**: the header splitter can change the text, so the position of a section cannot always be found back in the original file. Such sections are counted and reported.

## Resources

### References

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers), in particular text generation and chat templating
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Python Fire](https://github.com/google/python-fire)

### Use of AI

An AI assistant (Claude) was used during the project for:

- explaining concepts (tokenization, attention mask, chat templates, regular expressions, type hints);
- debugging specific errors (Fire argument parsing, pydantic validation, path handling, typing issues with `transformers`);
- reviewing code, suggesting fixes (for example the inverted overlap computation, the model loaded twice, and the missing parentheses in `is_file()`), and writing docstrings and formatting the code to the flake8 and mypy rules;
- Drafting this README.

The architecture, the implementation choices and the final code were reviewed, tested and adapted by the author, who is able to explain all of it.