# Dataset builder

## Purpose
- Build and preprocess image datasets for Trash Optimizer.
- Download sources (Kaggle / Hugging Face), map categories, apply augmentation, and export a timestamped processed dataset.

## Requirements
- Python 3.12 (pyenv recommended).
- Install component deps:
  - `cd data`
  - `pip install -r requirements.txt`
- Optional credentials:
  - Kaggle: `~/.kaggle/kaggle.json`

## Quick usage
- Edit the TOML config (default: `trash-optimizer-datasetbuilder.toml`) to set sources, category mappings and options.
- Run:
  - `export BUILDER_CONFIGURATION=trash-optimizer-datasetbuilder.toml`
  - `python datasetbuilder.py`

## Key configuration options
- `input_output_categories`: map source categories → standardized project categories.
- `output_augmentation`: `true`/`false` — enable augmentation.
- `output_max_per_category`: cap images per class.
- `output_minimum_images_size_wh`: minimum width/height for kept images.
- Source sections define Kaggle / HF dataset identifiers and download behavior.

## Output
- Processed datasets are written to:
  - `~/Data/trash-optimizer/datasets_processed/trash_optimizer_dataset_YYYYMMDD-HHMMSS/`
- Includes standardized folders per category and a manifest describing mappings and applied transforms.

## Notes / tips
- Ensure TOML mappings cover all source categories to avoid dropped images.
- Paths in config support `~` and are expanded with `Path.expanduser()`.
- Use augmentation and max-per-category to balance classes before training.
- See project-level docs for workflows that consume these datasets (training scripts expect the standardized layout).
