"""Configuration for evaluation runners; does not alter backend settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH", ROOT / "evals/datasets/evaluation_dataset.json"))
REPORT_DIR = Path(os.getenv("EVAL_REPORT_DIR", ROOT / "evals/reports"))
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "fi-bot-rag-evaluation")
LANGSMITH_DATASET = os.getenv("LANGSMITH_DATASET", "fi-bot-rag-evaluation-dataset")
