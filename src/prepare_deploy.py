"""Assemble a self-contained folder ready to push to Hugging Face Spaces.

    python -m src.prepare_deploy

The Space only ever *serves*; it never rebuilds anything. So it needs the fitted
models and a job table with the columns the UI reads — and nothing else. The
full `jobs.parquet` carries `description`, `job_text` and `job_text_clean`,
which together are 32MB of text used only when building the index and the
TF-IDF matrix. Dropping them takes the table from 18MB to 0.25MB and brings
every file under Hugging Face's 10MB limit, so the push needs no Git LFS.
"""

import json
import shutil

import pandas as pd

from src.config import (
    FAISS_INDEX_PATH,
    JOBS_META_PATH,
    JOBS_PARQUET,
    PROJECT_ROOT,
    RESULTS_DIR,
    SKILLS_TAXONOMY,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
)

DEPLOY_DIR = PROJECT_ROOT / "deploy"

# Columns the running app reads. Everything else is build-time only.
SERVING_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "skills",
    "min_years_experience",
]

SOURCE_MODULES = [
    "__init__.py",
    "config.py",
    "pdf_parser.py",
    "preprocessing.py",
    "resume_parser.py",
    "recommender.py",
    "embedder.py",
    "baseline_tfidf.py",
    "explainer_rules.py",
]

SPACE_HEADER = """---
title: NLP Job Recommender
emoji: 🔎
colorFrom: gray
colorTo: red
sdk: streamlit
sdk_version: 1.63.0
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
---

"""

# The dev and build-only dependencies are dropped: the Space never runs the
# test suite, the notebooks, or the corpus build.
SPACE_REQUIREMENTS = """# CPU-only torch keeps the Space image small and the build fast.
--extra-index-url https://download.pytorch.org/whl/cpu

streamlit==1.63.0
pandas==3.0.5
numpy==2.5.3
scikit-learn==1.9.1
sentence-transformers==6.0.1
faiss-cpu==1.15.0
spacy==3.8.16
pdfplumber==0.11.10
plotly==7.0.0
pyarrow==25.0.1
torch==2.14.0+cpu

en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
"""


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if DEPLOY_DIR.exists():
        shutil.rmtree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True)

    # Slim job table.
    jobs = pd.read_parquet(JOBS_PARQUET, columns=SERVING_COLUMNS)
    target = DEPLOY_DIR / "data" / "processed" / "jobs.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_parquet(target, index=False)

    # Models, taxonomy, app and source.
    _copy(SKILLS_TAXONOMY, DEPLOY_DIR / "data" / "skills_taxonomy.txt")
    for path in (FAISS_INDEX_PATH, JOBS_META_PATH, TFIDF_MATRIX_PATH, TFIDF_VECTORIZER_PATH):
        _copy(path, DEPLOY_DIR / "models" / path.name)

    _copy(PROJECT_ROOT / "app.py", DEPLOY_DIR / "app.py")
    _copy(PROJECT_ROOT / ".streamlit" / "config.toml", DEPLOY_DIR / ".streamlit" / "config.toml")
    for name in SOURCE_MODULES:
        _copy(PROJECT_ROOT / "src" / name, DEPLOY_DIR / "src" / name)

    # Benchmark results power the Evaluation tab.
    for name in ("benchmark.json", "ablation.json"):
        if (RESULTS_DIR / name).exists():
            _copy(RESULTS_DIR / name, DEPLOY_DIR / "results" / name)

    (DEPLOY_DIR / "requirements.txt").write_text(SPACE_REQUIREMENTS)

    readme = PROJECT_ROOT / "README.md"
    (DEPLOY_DIR / "README.md").write_text(SPACE_HEADER + readme.read_text())

    # Screenshots referenced by the README.
    for image in sorted((PROJECT_ROOT / "docs").glob("*.png")):
        _copy(image, DEPLOY_DIR / "docs" / image.name)

    files = sorted(path for path in DEPLOY_DIR.rglob("*") if path.is_file())
    total = sum(path.stat().st_size for path in files)
    largest = max(files, key=lambda path: path.stat().st_size)

    print(f"Wrote {DEPLOY_DIR}")
    print(f"  files:   {len(files)}")
    print(f"  total:   {total / 1e6:.1f} MB")
    print(f"  largest: {largest.relative_to(DEPLOY_DIR)} "
          f"({largest.stat().st_size / 1e6:.1f} MB)")
    if largest.stat().st_size > 10e6:
        print("  WARNING: a file exceeds Hugging Face's 10MB limit; Git LFS is needed.")
    else:
        print("  every file is under Hugging Face's 10MB limit — no Git LFS needed.")


if __name__ == "__main__":
    main()
