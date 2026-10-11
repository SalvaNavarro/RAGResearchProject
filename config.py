from pathlib import Path
import logging

# declare all to be able to import everything with *
__all__ = [
    "BASE",
    "DA_BASE",
    "DA_OUTPUT",
    "DA_TEXT",
    "DA_TEXT_IND",
    "DB_SETUP",
    "DB_SETUP_DATA",
    "APP",
    "APP_PROJECTS",
    "SERVICES",
    "SERVICES_RAG",
    "SERVICES_RAG_DATA",
    "SERVICES_RAG_DB",
    "SERVICES_RAG_DB_IMAGE",
    "SERVICES_RAG_DB_TEXT",
    "RAG_EVAL",
    "RAG_RETRIEVAL_EVAL_CSV",
    "RAG_GENERATION_EVAL_CSV",
    "RAG_EVAL_CSV",
    "RAG_BASELINE_EVAL_CSV",
    "RAG_BASELINE_DB",
    "RAG_BASELINE_DB_TEXT",
    "SUMMARY_FILE",
    "TEST_RAG_DB",
    "TEST_RAG_DATA",
    "TEST_RAG_DB_IMAGE",
    "TEST_RAG_DB_TEXT",
    "TEST_APP_PROJECTS",
    "RAG_BENCH_EVAL_DB",
    "RAG_BENCH_DB_TEXT",
    "RAG_BENCH_CSV",
    
    "test_logger",
    "rag_logger",
    "g_logger",
    "ignore_warnings"
]

# top level directory
BASE = Path(__file__).resolve().parent

# data pre-processing tool paths
DA_BASE = BASE / "utility" / "data_analysis"
DA_OUTPUT = DA_BASE / "output"
DA_TEXT = DA_BASE / "text"
DA_TEXT_IND = DA_TEXT / "individual"

# input data paths
DB_SETUP = BASE / "utility" / "db_setup"
DB_SETUP_DATA = BASE / "utility" / "db_setup" / "data"

# app paths
APP = BASE / "app"
APP_PROJECTS = APP / "static" / "projects"

# rag paths
SERVICES = BASE / "services"
SERVICES_RAG = SERVICES / "rag"
SERVICES_RAG_DATA = SERVICES_RAG / "data"
SERVICES_RAG_DB = SERVICES_RAG / "db"
SERVICES_RAG_DB_IMAGE = SERVICES_RAG_DB / "img"
SERVICES_RAG_DB_TEXT = SERVICES_RAG_DB / "txt" / "chunks.txt"

# RAG main evaluation results paths
RAG_EVAL = BASE / "utility" / "rag_evaluation"
RAG_RETRIEVAL_EVAL_CSV = RAG_EVAL / "rag_retrieval_eval_results.csv"
RAG_GENERATION_EVAL_CSV = RAG_EVAL / "rag_generation_eval_results.csv"
RAG_EVAL_CSV = RAG_EVAL / "csv_full_evaluation"

# RAG baseline evaluation results paths
RAG_BASELINE_DB = RAG_EVAL / "baseline_rag_db"
RAG_BASELINE_DB_TEXT = RAG_BASELINE_DB / "txt" / "chunks_bench.txt"
RAG_BASELINE_EVAL_CSV = RAG_EVAL / "csv_baseline"

# relevant links for benchmarking with other known public datasets
RAG_BENCH_EVAL_DB = RAG_EVAL / "bench_rag_db"
RAG_BENCH_DB_TEXT = RAG_BENCH_EVAL_DB / "txt" / "chunks_bench.txt"
RAG_BENCH_CSV = RAG_EVAL / "csv_benchmarking"

# test suite paths holding smaller subsets of the data
TEST_RAG_DB = BASE / "tests" / "test_rag_db"
TEST_RAG_DATA = TEST_RAG_DB / "data"
TEST_RAG_DB_IMAGE = TEST_RAG_DB / "img"
TEST_RAG_DB_TEXT = TEST_RAG_DB / "txt" / "chunks_test.txt"
TEST_APP_PROJECTS = TEST_RAG_DATA / "projects"


# standard file names
SUMMARY_FILE = "summary.txt"

# logging settings for tests
logging.basicConfig(level=logging.WARNING)
test_logger = logging.getLogger(__name__) 
test_logger.setLevel(level=logging.INFO)

# logging RAG steps
logging.basicConfig(level=logging.WARNING)
rag_logger = logging.getLogger("rag") 
rag_logger.setLevel(level=logging.INFO)

# general / multipurpose logger
logging.basicConfig(level=logging.WARNING)
g_logger = logging.getLogger("general") 
g_logger.setLevel(level=logging.INFO)

# ignore warnings
def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore")   

# flask app factory configurations
class BaseConfig:
    SQLALCHEMY_TRACK_MODIFICATIONS = False # for dev stage only
    TESTING = False

class ProductionConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = "sqlite:///production.db"

class TestingConfig(BaseConfig):
    SQLALCHEMY_DATABASE_URI = "sqlite:///test.db" # temp database in memory
    TESTING = True