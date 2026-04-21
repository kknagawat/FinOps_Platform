"""
run_step4.py
============
Step 4 — Multi-Tool LangChain Agent Runner
-------------------------------------------
Initialises the FinOpsAgent and runs it against all 12 required test questions.
Saves full results to agent_evaluation.json for submission.

Prerequisites
-------------
  1. finops.db must exist  (python run_step2.py)
  2. ANTHROPIC_API_KEY must be set:
       export ANTHROPIC_API_KEY=sk-ant-...
     Or place it in a .env file at the project root:
       ANTHROPIC_API_KEY=sk-ant-...

Usage
-----
  python run_step4.py                     # run all 12 test questions
  python run_step4.py --question 4        # run a single question by number (1-12)
  python run_step4.py --interactive       # open chat mode for free-form questions
  python run_step4.py --verbose           # show full chain-of-thought from agent
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy HTTP / LangChain logs; keep WARNING and above
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s │ %(message)s")
for noisy in ("httpx", "anthropic", "langchain", "langchain_core", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


# =============================================================================
# The 12 required test questions
# =============================================================================

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is our current MRR and how has it changed over the last 6 months?",
        "expected_tools": ["revenue_calculator_tool"],
        "routing_note":   "Revenue metric → revenue_calculator_tool",
    },
    {
        "id": 2,
        "question": "Show me the top 10 customers by lifetime revenue.",
        "expected_tools": ["sql_query_tool"],
        "routing_note":   "Data lookup → sql_query_tool",
    },
    {
        "id": 3,
        "question": "Which customers are at high risk of churning based on recent behavior?",
        "expected_tools": ["customer_segmentation_tool"],
        "routing_note":   "Churn risk segmentation → customer_segmentation_tool",
    },
    {
        "id": 4,
        "question": "What is our refund policy for enterprise customers?",
        "expected_tools": ["knowledge_retrieval_tool"],
        "routing_note":   "Policy question → MUST use knowledge_retrieval_tool",
    },
    {
        "id": 5,
        "question": "Generate a bar chart of monthly revenue for the last 12 months.",
        "expected_tools": ["chart_generator_tool"],
        "routing_note":   "Visualisation → chart_generator_tool",
    },
    {
        "id": 6,
        "question": "Are there any anomalies in our transaction volumes this month?",
        "expected_tools": ["anomaly_detection_tool"],
        "routing_note":   "Anomaly detection → anomaly_detection_tool",
    },
    {
        "id": 7,
        "question": "Segment our customers using RFM analysis and show the distribution.",
        "expected_tools": ["customer_segmentation_tool"],
        "routing_note":   "RFM segmentation → customer_segmentation_tool",
    },
    {
        "id": 8,
        "question": (
            "What is the average resolution time for high-priority support tickets, "
            "and what does our SLA require?"
        ),
        "expected_tools": ["sql_query_tool", "knowledge_retrieval_tool"],
        "routing_note":   "Multi-step: data lookup + policy retrieval",
    },
    {
        "id": 9,
        "question": (
            "Show me the revenue cohort retention curve for customers "
            "who signed up in January."
        ),
        "expected_tools": ["sql_query_tool", "chart_generator_tool"],
        "routing_note":   "Multi-step: cohort SQL + chart generation",
    },
    {
        "id": 10,
        "question": "Which product features have the lowest adoption rate?",
        "expected_tools": ["sql_query_tool"],
        "routing_note":   "Feature adoption → sql_query_tool",
    },
    {
        "id": 11,
        "question": (
            "Find customers with overlapping subscriptions and calculate "
            "the double-billing impact."
        ),
        "expected_tools": ["sql_query_tool"],
        "routing_note":   "Overlap detection (Q4 SQL pattern) → sql_query_tool",
    },
    {
        "id": 12,
        "question": (
            "Compare the churn rate between customers on the Pro plan vs. Basic plan, "
            "and forecast the trend for the next quarter."
        ),
        "expected_tools": ["sql_query_tool", "forecast_tool"],
        "routing_note":   "Multi-step: plan comparison + forecast",
    },
]


# =============================================================================
# Helpers
# =============================================================================

def _load_env() -> None:
    """Load ANTHROPIC_API_KEY from .env if not already in environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    os.environ["ANTHROPIC_API_KEY"] = key
                    return


def _check_prerequisites() -> None:
    _load_env()
    db = PROJECT_ROOT / "finops.db"
    if not db.exists():
        print("✗  finops.db not found.")
        print("   Run: python run_step2.py")
        sys.exit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗  ANTHROPIC_API_KEY not set.")
        print("   Run: export ANTHROPIC_API_KEY=sk-ant-...")
        print("   Or add it to .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)


def _routing_status(expected: list[str], used: list[str]) -> tuple[str, str]:
    exp_set  = set(expected)
    used_set = set(used)
    if exp_set.issubset(used_set):
        return "✓ Correct", "correct"
    if exp_set & used_set:
        return "~ Partial", "partial"
    return "✗ Missed", "missed"


def _print_result(q_meta: dict, result: dict, total: int) -> None:
    SEP  = "─" * 72
    SEP2 = "═" * 72
    status_label, _ = _routing_status(q_meta["expected_tools"], result["tools_used"])

    print(f"\n{SEP2}")
    print(f"  Q{q_meta['id']:02d}/{total}  {q_meta['question']}")
    print(f"  Expected : {q_meta['expected_tools']}")
    print(f"  Used     : {result['tools_used']}")
    print(f"  Routing  : {status_label}  │  {result['execution_time_ms']:.0f} ms  │  confidence {result['confidence']:.0%}")
    print(SEP)
    answer = result["answer"]
    if len(answer) > 700:
        answer = answer[:697] + "..."
    print(f"\n{answer}\n")
    if result.get("generated_sql"):
        preview = result["generated_sql"].replace("\n", " ")[:120]
        print(f"  SQL      : {preview}…")
    if result.get("chart_path"):
        print(f"  Chart    : {result['chart_path']}")
    if result.get("sources"):
        for s in result["sources"]:
            print(f"  Source   : {s.get('source','')} — {s.get('section','')}")


def _save_evaluation(all_results: list[dict]) -> Path:
    out_path = PROJECT_ROOT / "agent_evaluation.json"
    records  = []
    for item in all_results:
        q = item["meta"]
        r = item["result"]
        status_label, status_key = _routing_status(q["expected_tools"], r["tools_used"])
        records.append({
            "id":                q["id"],
            "question":          q["question"],
            "expected_tools":    q["expected_tools"],
            "actual_tools":      r["tools_used"],
            "routing_status":    status_key,
            "answer_summary":    r["answer"][:500],
            "generated_sql":     r.get("generated_sql"),
            "chart_path":        r.get("chart_path"),
            "sources":           r.get("sources"),
            "execution_time_ms": r["execution_time_ms"],
            "confidence":        r["confidence"],
        })
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return out_path


# =============================================================================
# Runners
# =============================================================================

def run_all(agent, questions: list[dict]) -> list[dict]:
    n           = len(questions)
    all_results = []
    for q_meta in questions:
        print(f"\n  Running Q{q_meta['id']:02d}/{n}…", end="", flush=True)
        result = agent.query(q_meta["question"])
        print(f" done ({result['execution_time_ms']:.0f} ms)")
        _print_result(q_meta, result, n)
        all_results.append({"meta": q_meta, "result": result})
    return all_results


def run_interactive(agent) -> None:
    print("\n  Interactive mode — type your question, 'reset' to clear memory, 'exit' to quit.\n")
    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            agent.reset_memory()
            print("  Memory cleared.\n")
            continue
        result = agent.query(question)
        print(f"\n  Agent: {result['answer']}")
        if result["tools_used"]:
            print(f"  [tools: {', '.join(result['tools_used'])} | {result['execution_time_ms']:.0f} ms]\n")


def _print_scorecard(all_results: list[dict]) -> None:
    SEP2 = "═" * 72
    print(f"\n{SEP2}")
    print("  EVALUATION SCORECARD")
    print(SEP2)
    correct = partial = wrong = 0
    total_ms = 0.0
    print(f"\n  {'Q':>3}  {'Expected':<40} {'Status':<12}  {'ms':>6}")
    print("  " + "─" * 65)
    for item in all_results:
        q = item["meta"]
        r = item["result"]
        label, key = _routing_status(q["expected_tools"], r["tools_used"])
        if key == "correct": correct += 1
        elif key == "partial": partial += 1
        else: wrong += 1
        total_ms += r["execution_time_ms"]
        print(f"  Q{q['id']:02d}  {str(q['expected_tools']):<40} {label:<12} {r['execution_time_ms']:>6.0f}")
    n = len(all_results)
    print(f"\n  Correct routing  : {correct}/{n}")
    print(f"  Partial routing  : {partial}/{n}")
    print(f"  Missed routing   : {wrong}/{n}")
    print(f"  Average latency  : {total_ms/n:.0f} ms / question")
    print(f"  Total time       : {total_ms/1000:.1f} s")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 — FinOps Agent Runner")
    parser.add_argument("--question",    type=int, help="Run one question by number (1-12)")
    parser.add_argument("--interactive", action="store_true", help="Open interactive chat")
    parser.add_argument("--verbose",     action="store_true", help="Show agent chain-of-thought")
    args = parser.parse_args()

    _check_prerequisites()

    BANNER = "█" * 72
    print()
    print(BANNER)
    print("  FINOPS ANALYTICS PLATFORM — STEP 4")
    print("  Multi-Tool LangChain Agent  (Claude claude-opus-4-5)")
    print(BANNER)

    print("\n  Initialising agent…", end="", flush=True)
    from src.agent.agent import FinOpsAgent
    agent = FinOpsAgent(verbose=args.verbose)
    print(f" ready\n  Tools: {agent.tool_names()}\n")

    if args.interactive:
        run_interactive(agent)
        return

    if args.question:
        matches = [q for q in TEST_QUESTIONS if q["id"] == args.question]
        if not matches:
            print(f"  No question #{args.question}. Valid: 1–12")
            sys.exit(1)
        result = agent.query(matches[0]["question"])
        _print_result(matches[0], result, 1)
        return

    # Run all 12
    print(f"  Running all {len(TEST_QUESTIONS)} test questions…")
    all_results = run_all(agent, TEST_QUESTIONS)
    _print_scorecard(all_results)
    out = _save_evaluation(all_results)
    print(f"\n  Evaluation file  : {out}")
    print("═" * 72)


if __name__ == "__main__":
    main()