# IssueExec (Anonymous Artifact)

IssueExec is a test-driven framework for **software issue localization**. It operationalizes the paper’s idea that **tests serve as executable requirements** by (1) retrieving requirement-relevant tests, (2) analyzing execution traces to identify suspicious code locations, (3) refining candidates using repository structure and supplemental retrieval, and (4) reranking to output final ranked edit locations.

This repository is prepared to meet **top-tier double-blind review** artifact standards:
- No author / affiliation identifiers
- No private links, tokens, or machine-specific paths committed
- Reproducible, CLI-driven execution with explicit inputs/outputs

---

## Repository Structure

- `localize.py` — main pipeline entry (multi-stage localization).
- `Localizer.py` — core localization components.
- `prompt.py` — prompt templates.
- `merge.py` — merges outputs from multiple stages.
- `util/` — shared utilities (pre/post-processing, model/API wrappers, structure parsing).

---

## Environment Setup (Required)

### 1) Install Dependencies

```bash
pip install -r requirements.txt
````

### 2) Configure API Keys

Set the model backend endpoint and key (example uses OpenAI-compatible interface):

```bash
export OPENAI_BASE_URL="<openai_base_url>"
export OPENAI_API_KEY="<your_api_key>"
```

> **Note:** Do not hardcode or commit any secrets.

---

## Data Preparation (Required)

### 3) Unpack Example Data

```bash
tar -xzf example_data.tar.gz
```

This creates the example dataset and required auxiliary files (e.g., coverage graph).

---

## Running the Pipeline (All Steps)

All commands below are **required** to reproduce the full pipeline described in the execution document.

### 4) Relevant Test Retrieval

```bash
python localize.py \
    --stage related_tests_retrieval \
    --output_folder example \
    --num_threads 2 \
    --skip_existing \
    --model gpt-4o-2024-05-13 \
    --backend openai \
    --top_n 5 \
    --dataset example_data/issues/test \
    --coverage_graph_path example_data/coverage_graph
```

* Produces initial localization outputs under:

  * `example/related_tests_retrieval/loc_outputs.jsonl`

---

### 5) Trace-Guided Localization

```bash
python localize.py \
    --stage blind_spot_analysis \
    --output_folder example \
    --num_threads 2 \
    --skip_existing \
    --start_file example/related_tests_retrieval/loc_outputs.jsonl \
    --model gpt-4o-2024-05-13 \
    --backend openai \
    --dataset example_data/issues/test
```

* Consumes the previous stage output via `--start_file`.

---

### 6) Refinement (Supplementary Retrieval)

```bash
python localize.py \
    --stage suppletory_retrieval \
    --output_folder example \
    --num_threads 2 \
    --skip_existing \
    --start_file example/blind_spot_analysis/loc_outputs.jsonl \
    --model gpt-4o-2024-05-13 \
    --backend openai \
    --dataset example_data/issues/test
```

---

### 7) Merge Outputs

```bash
python merge.py --target_folder example
```

* Produces merged outputs under:

  * `example/merge/loc_outputs.jsonl`

---

### 8) Reranking

```bash
python localize.py \
  --stage reranking \
  --output_folder example \
  --num_threads 2 \
  --skip_existing \
  --start_file example/merge/loc_outputs.jsonl \
  --model gpt-4o-2024-05-13 \
  --backend openai \
  --context_expansion \
  --dataset example_data/issues/test
```

* This final stage outputs the ranked edit locations for downstream patch generation.