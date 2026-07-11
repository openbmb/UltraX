import os

# ============================================================
# API Configuration
# ============================================================
API_URL = os.environ.get("API_URL", "")
API_KEY = os.environ.get("API_KEY", "")

REFINE_MODEL = os.environ.get("REFINE_MODEL", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")

REFINE_TEMPERATURE = 0
JUDGE_TEMPERATURE = 0.4
API_MAX_TOKENS = 24000
API_TIMEOUT = 180
API_RETRY_TIMES = 5
API_RETRY_DELAY = 5
API_CONCURRENCY = 5
API_CONCURRENCY_TEST = 30

# ============================================================
# Data Paths
# ============================================================
BASE_DIR = os.environ.get("BASE_DIR", "")
SEED_DATA_DIR = os.path.join(BASE_DIR, "seed_data")
PROMPT_REFERENCE_DIR = os.path.join(BASE_DIR, "prompt_reference")
BASE_PROMPT_FILE = os.environ.get(
    "BASE_PROMPT_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinement_prompt_en.txt"),
)
OUTPUT_DIR = os.path.join(BASE_DIR, "prompt_optimizer_outputs")

# ============================================================
# Sampling
# ============================================================
OPT_SAMPLE_SIZE = 1000
TEST_SAMPLE_SIZE = 1000
TEXT_COLUMN = "content"

# ============================================================
# Optimization Loop
# ============================================================
MAX_ITERATIONS = 200
REFINE_BATCH_SIZE = 5
MAX_INNER_RETRIES = 5
