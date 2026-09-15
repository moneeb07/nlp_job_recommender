"""Central configuration: every file path and tunable constant lives here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVAL_DATA_DIR = DATA_DIR / "eval"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_POSTINGS_CSV = RAW_DATA_DIR / "postings.csv"
JOBS_PARQUET = PROCESSED_DATA_DIR / "jobs.parquet"
SKILLS_TAXONOMY = DATA_DIR / "skills_taxonomy.txt"
EVAL_SET_JSON = EVAL_DATA_DIR / "eval_set.json"

FAISS_INDEX_PATH = MODELS_DIR / "jobs.faiss"
JOBS_META_PATH = MODELS_DIR / "jobs_meta.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"
TFIDF_META_PATH = MODELS_DIR / "tfidf_meta.json"

SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

SAMPLE_SIZE = 5000
MIN_DESCRIPTION_CHARS = 100
JOB_TEXT_DESCRIPTION_CHARS = 1500
RANDOM_SEED = 42

TOP_K = 10
RETRIEVE_TOP_N = 100

WEIGHT_SEMANTIC = 0.50
WEIGHT_SKILL_OVERLAP = 0.30
WEIGHT_EXPERIENCE = 0.20

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 50_000
TFIDF_SUBLINEAR_TF = True

STRONG_FIT_THRESHOLD = 0.6
GOOD_FIT_THRESHOLD = 0.3

MAX_TOKEN_ROWS = 3000

for _directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EVAL_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)
