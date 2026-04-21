"""
run_step2.py  —  Step 2: Schema Design & SQLite Loading
Run: python run_step2.py
Requires: run_all.py to have been run first (or runs it automatically).
"""
import sys, time, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

from src.etl.ingestion import run_ingestion
from src.etl.cleaning  import run_cleaning
from src.etl.loader    import run_loader, print_load_summary

def main():
    print("\n" + "="*60)
    print("  STEP 2 — SQLite Schema Design & Loading")
    print("="*60 + "\n")
    t = time.perf_counter()

    print("--- Running Step 1 first (ingestion + cleaning) ---\n")
    raw     = run_ingestion()
    cleaned = run_cleaning(raw)

    print("\n--- Loading into SQLite ---\n")
    result = run_loader(cleaned)
    print_load_summary(result)

    elapsed = time.perf_counter() - t
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Database: finops.db\n")

if __name__ == "__main__":
    main()