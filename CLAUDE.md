# CLAUDE.md — Chandra OCR

This file provides guidance for AI assistants working on this codebase.

## Project Overview

**Chandra OCR** (`chandra-ocr` v0.2.0) is a state-of-the-art OCR system that converts images and PDFs into structured Markdown, HTML, or JSON while preserving layout metadata. It supports 90+ languages, math/equations, tables, forms, and handwriting.

Built by [Datalab](https://datalab.to). Apache-2.0 licensed code; model weights use OpenRAIL-M.

---

## Repository Structure

```
chandra/               # Main Python package
├── __init__.py
├── input.py           # PDF/image loading (pypdfium2, PIL)
├── output.py          # HTML/Markdown parsing and output formatting
├── prompts.py         # OCR prompt templates and allowed HTML tags
├── settings.py        # Pydantic settings (env-var driven)
├── util.py            # Layout visualization utilities
├── model/             # Inference backends
│   ├── __init__.py    # InferenceManager (factory for hf/vllm)
│   ├── hf.py          # HuggingFace Transformers backend
│   ├── vllm.py        # Remote vLLM backend (OpenAI-compatible API)
│   ├── schema.py      # Data models: BatchInputItem, BatchOutputItem, GenerationResult
│   └── util.py        # Image scaling, repeat token detection
└── scripts/           # CLI and app entry points
    ├── cli.py          # Main `chandra` CLI (click-based)
    ├── app.py          # Streamlit web app
    ├── run_app.py      # Web app entry point
    ├── vllm.py         # vLLM Docker server launcher
    ├── screenshot_app.py  # Flask app for layout visualization
    └── templates/
        └── screenshot.html  # HTML template for screenshot app
tests/
├── conftest.py         # Pytest fixtures
└── integration/
    └── test_image_inference.py
.github/workflows/
├── integration.yml     # CI: runs integration tests on GPU runner
└── publish.yml         # CD: builds and publishes to PyPI on version tags
```

---

## Development Setup

**Python**: 3.12 (see `.python-version`). Requires Python >=3.10.

**Package manager**: [`uv`](https://github.com/astral-sh/uv) — do not use pip directly.

```bash
# Install all dependencies (including dev extras)
uv sync --group dev

# Activate virtual environment
source .venv/bin/activate
```

**Optional dependency groups:**
- `hf` — local HuggingFace inference (torch, torchvision, transformers, accelerate)
- `app` — Streamlit web UI
- `all` — hf + app

```bash
uv sync --extra all --group dev
```

---

## Running Tests

Tests are integration tests that require GPU and a valid `HF_TOKEN`.

```bash
# Run integration tests
HF_TOKEN=<token> TORCH_ATTN=sdpa uv run pytest tests/integration

# Run with explicit PYTHONPATH
PYTHONPATH=. HF_TOKEN=<token> TORCH_ATTN=sdpa uv run pytest tests/integration
```

CI runs on a GPU runner (`t4_gpu`). Do not expect tests to pass locally without a GPU and HuggingFace credentials.

---

## Code Style and Linting

**Linter/Formatter**: [Ruff](https://docs.astral.sh/ruff/) v0.14.1 via pre-commit hooks.

```bash
# Install pre-commit hooks (run once)
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files

# Or run ruff directly
uv run ruff check --fix .
uv run ruff format .
```

No mypy or other type checkers are configured. Type hints are used throughout but not enforced by CI.

---

## Configuration

Settings are managed via `chandra/settings.py` using Pydantic `BaseSettings`. All settings can be overridden via environment variables or a `local.env` file in the project root.

| Setting | Default | Description |
|---|---|---|
| `MODEL_CHECKPOINT` | `datalab-to/chandra-ocr-2` | HuggingFace model |
| `IMAGE_DPI` | `192` | PDF render DPI |
| `MIN_PDF_IMAGE_DIM` | `1024` | Min PDF image dimension |
| `MIN_IMAGE_DIM` | `1536` | Min standalone image dimension |
| `MAX_OUTPUT_TOKENS` | `12384` | Token limit per page |
| `TORCH_DEVICE` | `None` (auto) | Force a specific torch device |
| `TORCH_ATTN` | `None` | Attention implementation (`sdpa`, `flash_attention_2`) |
| `BBOX_SCALE` | `1000` | Bounding box coordinate scale |
| `VLLM_API_BASE` | `http://localhost:8000/v1` | vLLM server URL |
| `VLLM_API_KEY` | `EMPTY` | vLLM API key |
| `VLLM_MODEL_NAME` | `chandra` | Model name served by vLLM |
| `VLLM_GPUS` | `0` | GPU device IDs for vLLM server |
| `MAX_VLLM_RETRIES` | `6` | Retry limit for vLLM requests |

---

## Data Models

All inter-module data exchange uses schema classes from `chandra/model/schema.py`:

- **`BatchInputItem`** — input to inference: `image` (PIL Image), `prompt` (optional override), `prompt_type` (`"ocr_layout"` or `"ocr"`)
- **`GenerationResult`** — raw inference output: `raw` (HTML string), `token_count`, `error` (bool)
- **`BatchOutputItem`** — fully processed output: `markdown`, `html`, `chunks` (layout blocks dict), `raw`, `page_box`, `token_count`, `images` (dict of PIL Images), `error`

---

## Inference Backends

`InferenceManager` (in `chandra/model/__init__.py`) abstracts two backends:

### HuggingFace (`--method hf`)
- Loads `AutoModelForImageTextToText` locally
- Suitable for single-GPU development
- Default batch size: 1

### vLLM (`--method vllm`, default)
- Calls an OpenAI-compatible HTTP API
- Suitable for production / high-throughput
- Default batch size: 28
- Requires a running vLLM server (see `chandra_vllm`)
- Uses parallel `ThreadPoolExecutor` with retry logic and repeat-token detection

---

## CLI Usage

```bash
# Basic OCR (vLLM backend, default)
chandra <input_path> <output_path>

# Use local HuggingFace model
chandra <input_path> <output_path> --method hf

# Process specific PDF pages
chandra doc.pdf output/ --page-range 1-5,7,9-12

# Control output
chandra doc.pdf output/ --no-images --no-html --no-chunks

# Parallel vLLM processing
chandra docs/ output/ --max-workers 8 --batch-size 28

# Paginate output (add page separators in Markdown/HTML)
chandra doc.pdf output/ --paginate-output
```

**All CLI options:**

| Option | Default | Description |
|---|---|---|
| `--method [hf\|vllm]` | `vllm` | Inference backend |
| `--page-range TEXT` | all pages | Pages to process, e.g. `1-5,7,9-12` |
| `--max-output-tokens INT` | from settings | Token limit per page |
| `--max-workers INT` | 4 | Parallel workers (vLLM only) |
| `--max-retries INT` | from settings | Retry limit (vLLM only) |
| `--batch-size INT` | 1 (hf) / 28 (vllm) | Pages per batch |
| `--include-images/--no-images` | include | Extract and save figure images |
| `--include-headers-footers/--no-headers-footers` | exclude | Include page headers/footers |
| `--save-html/--no-html` | save | Write `.html` output file |
| `--save-chunks/--no-chunks` | save | Write `_chunks.json` output file |
| `--paginate-output` | off | Add page separators to output |

Output per file is saved to `<output_path>/<stem>/`:
- `<stem>.md` — Markdown
- `<stem>.html` — HTML (unless `--no-html`)
- `<stem>_chunks.json` — layout blocks with bounding boxes (unless `--no-chunks`)
- `<stem>_metadata.json` — page counts, token counts, image counts
- Extracted images as `.webp` files (unless `--no-images`)

---

## Data Flow

```
Input (PDF/image)
  → input.py: load_file()          # Render pages to PIL Images
  → BatchInputItem(image, prompt)
  → InferenceManager.generate()    # Run model (hf or vllm)
  → GenerationResult(raw HTML)
  → output.py: parse_markdown/html/chunks/extract_images
  → BatchOutputItem(markdown, html, chunks, images, metadata)
  → cli.py: save_merged_output()   # Write files to disk
```

---

## Output Format

The model outputs structured HTML with layout blocks. Each block has:
- A semantic tag (paragraph, heading, table, equation, figure, etc.)
- A `data-bbox` attribute with normalized coordinates `[x0, y0, x1, y1]` scaled to `BBOX_SCALE` (default 1000)
- A `data-label` attribute identifying the block type

**Supported layout labels (17 types):** `Caption`, `Footnote`, `Equation-Block`, `List-Group`, `Page-Header`, `Page-Footer`, `Image`, `Section-Header`, `Table`, `Text`, `Complex-Block`, `Code-Block`, `Form`, `Table-Of-Contents`, `Figure`, `Chemical-Block`, `Diagram`, `Bibliography`, `Blank-Page`

`output.py` parses this HTML into:
- Clean Markdown (via custom `Markdownify` subclass with math/table support)
- Filtered HTML (optional header/footer removal, image toggling)
- `chunks` — list of layout blocks with bounding boxes (saved as `_chunks.json`)
- `images` — extracted figure images as PIL objects (saved as `.webp`)

---

## Prompts

Two prompt types in `chandra/prompts.py`:
- `ocr_layout` — full layout-aware OCR with bounding boxes (default)
- `ocr` — plain OCR without layout metadata

Prompt type is set per `BatchInputItem` and selects the template at inference time.

---

## Web Applications

### Streamlit App (`chandra_app`)
Interactive single-page OCR UI. Supports file upload, page selection, real-time processing, and Markdown download. Launched via `chandra_app` or `uv run python -m chandra.scripts.run_app`.

### Screenshot App (`chandra_screenshot`)
Flask-based layout visualization server. Renders color-coded layout blocks overlaid on the original image with side-by-side extracted text. REST API at `/process`. Useful for debugging and demos.

### vLLM Server (`chandra_vllm`)
Docker-based vLLM server launcher. Automatically scales configuration (max batch tokens, sequence limits) based on GPU VRAM. Supports H100, A100, L40s, A10, L4, RTX4090, RTX3090, T4. Configure GPU selection via `VLLM_GPUS`.

---

## Adding a New Inference Backend

1. Create `chandra/model/<backend>.py` implementing a `generate_<backend>(batch, ...) -> List[GenerationResult]` function.
2. Add the backend option to `InferenceManager.__init__` and `InferenceManager.generate` in `chandra/model/__init__.py`.
3. Expose the new method in the CLI `--method` option in `chandra/scripts/cli.py`.

---

## CI/CD

### Integration Tests (`integration.yml`)
- Trigger: every push
- Runner: `t4_gpu` (GPU required)
- Installs system libs: `libpango`, `libharfbuzz`, `libcairo2`, `libgdk-pixbuf2.0`, `libffi-dev`
- Runs: `uv run pytest tests/integration`
- Env secrets: `HF_TOKEN`

### Publishing (`publish.yml`)
- Trigger: git tags matching `v*.*.*`
- Validates tag matches `version` in `pyproject.toml`
- Builds with `uv build`, publishes to PyPI with `uv publish`
- Secrets: `PYPI_TOKEN`

To release: bump `version` in `pyproject.toml`, commit, then push a tag `vX.Y.Z`.

---

## Key Conventions

- **No direct pip usage** — always use `uv`.
- **Settings via environment** — never hardcode credentials or paths; use `settings.py`.
- **Backends are stateless functions** — `generate_hf` and `generate_vllm` take a batch list and return results; side effects are minimal.
- **Schema-driven I/O** — all inter-module data uses `BatchInputItem`, `BatchOutputItem`, and `GenerationResult` from `chandra/model/schema.py`.
- **Output parsing is separate from inference** — `output.py` handles all HTML/Markdown transformation; inference code only returns raw HTML strings.
- **Ruff enforced** — run `pre-commit run --all-files` before committing.
- **Images saved as `.webp`** — extracted figure images use WebP format, not PNG.
