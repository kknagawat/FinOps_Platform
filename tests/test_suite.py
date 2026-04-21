"""
tests/test_suite.py
===================
Step 8 — Complete Test Suite

Unit tests
----------
  TestStripCurrency        — cleaning.strip_currency()
  TestParseDates           — cleaning.parse_dates_to_utc()
  TestNormaliseRating      — cleaning.normalise_rating()
  TestNormaliseBoolean     — cleaning.normalise_boolean()
  TestRemoveHtmlMarkdown   — cleaning.strip_html_markdown()
  TestSafetyClassifier     — classifier.classify()
  TestPIIMasking           — guardrails.mask_pii_in_text / mask_pii_in_value
  TestRowLimit             — guardrails.enforce_row_limit()
  TestReflection           — guardrails.reflect()
  TestRevenueCalculator    — tools.revenue_calculator_tool (mocked DataFrames)
  TestAnomalyDetection     — tools.anomaly_detection_tool (mocked DataFrames)
  TestApprovalQueue        — queue.enqueue / approve / reject lifecycle
  TestAuditLog             — audit.log_interaction / get_history

Integration tests
-----------------
  TestFastAPIEndpoints     — FastAPI TestClient against all key endpoints

Run
---
  pytest tests/test_suite.py -v
  pytest tests/test_suite.py -v -k "TestClassifier"   # single class
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock LangChain so tools.py can be imported without a real API key
# Only mock what actually needs mocking (LLM client and agent framework)
# langchain_core.tools must NOT be mocked - @tool decorator needs to work
for _mod in [
    "langchain_anthropic",
    "langgraph", "langgraph.prebuilt",
    "langgraph.checkpoint", "langgraph.checkpoint.memory",
]:
    sys.modules.setdefault(_mod, mock.MagicMock())


# =============================================================================
# Unit tests — ETL cleaning functions
# =============================================================================

class TestStripCurrency:
    """Tests for cleaning.strip_currency()"""

    def test_dollar_prefix(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["$100.50", "$250.00"])
        r = strip_currency(s)
        assert r[0] == pytest.approx(100.50)
        assert r[1] == pytest.approx(250.00)

    def test_usd_word_prefix(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["USD 507.90", "AED 365.94"])
        r = strip_currency(s)
        assert r[0] == pytest.approx(507.90)
        assert r[1] == pytest.approx(365.94)

    def test_currency_with_space(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["$ 826.53"])
        r = strip_currency(s)
        assert r[0] == pytest.approx(826.53)

    def test_plain_number(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["999.0"])
        r = strip_currency(s)
        assert r[0] == pytest.approx(999.0)

    def test_negative_preserved(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["-50.00"])
        r = strip_currency(s)
        assert r[0] == pytest.approx(-50.0)

    def test_invalid_becomes_nan(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series(["not_a_number"])
        r = strip_currency(s)
        assert np.isnan(r[0])

    def test_null_becomes_nan(self):
        from src.etl.cleaning import strip_currency
        s = pd.Series([None])
        r = strip_currency(s)
        assert np.isnan(r[0])


class TestParseDates:
    """Tests for cleaning.parse_dates_to_utc()"""

    def test_iso_format(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series(["2023-01-15"])
        assert parse_dates_to_utc(s)[0] == "2023-01-15"

    def test_slash_dd_mm_yyyy(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series(["15/06/2022"])
        assert parse_dates_to_utc(s)[0] == "2022-06-15"

    def test_natural_language(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series(["May 15, 2024"])
        assert parse_dates_to_utc(s)[0] == "2024-05-15"

    def test_iso_with_timezone(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series(["2023-06-24T00:00:00+00:00"])
        assert parse_dates_to_utc(s)[0] == "2023-06-24"

    def test_null_returns_none(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series([None])
        result = parse_dates_to_utc(s)[0]
        assert result is None or pd.isna(result)

    def test_slash_yyyy_mm_dd(self):
        from src.etl.cleaning import parse_dates_to_utc
        s = pd.Series(["2022/12/18"])
        assert parse_dates_to_utc(s)[0] == "2022-12-18"


class TestNormaliseRating:
    """Tests for cleaning.normalise_rating()"""

    def test_integer_string(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["4", "5", "1"])
        r = normalise_rating(s)
        assert r[0] == 4.0 and r[1] == 5.0 and r[2] == 1.0

    def test_word_form(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["four", "five", "one", "three"])
        r = normalise_rating(s)
        assert list(r) == [4.0, 5.0, 1.0, 3.0]

    def test_fraction_x_over_5(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["4/5"])
        r = normalise_rating(s)
        assert r[0] == pytest.approx(4.0, abs=0.1)

    def test_float_string(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["4.0", "3.5"])
        r = normalise_rating(s)
        assert r[0] == 4.0 and r[1] == 3.5

    def test_clamp_above_5(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["7"])
        r = normalise_rating(s)
        assert r[0] == 5.0

    def test_invalid_becomes_nan(self):
        from src.etl.cleaning import normalise_rating
        s = pd.Series(["excellent"])
        r = normalise_rating(s)
        assert np.isnan(r[0])


class TestNormaliseBoolean:
    """Tests for cleaning.normalise_boolean()"""

    def test_truthy_variants(self):
        from src.etl.cleaning import normalise_boolean
        for val in ["true", "True", "yes", "1", "Y", True, 1]:
            s = pd.Series([val])
            assert normalise_boolean(s)[0] == True, f"Failed for: {val}"

    def test_falsy_variants(self):
        from src.etl.cleaning import normalise_boolean
        for val in ["false", "False", "no", "0", "N", False, 0]:
            s = pd.Series([val])
            assert normalise_boolean(s)[0] == False, f"Failed for: {val}"

    def test_null_becomes_nan(self):
        from src.etl.cleaning import normalise_boolean
        s = pd.Series([None])
        r = normalise_boolean(s)
        assert np.isnan(r[0])


class TestStripHtmlMarkdown:
    """Tests for cleaning.strip_html_markdown()"""

    def test_html_tags_removed(self):
        from src.etl.cleaning import strip_html_markdown
        s = pd.Series(["<div>Fixed the <a href='#'>issue</a></div>"])
        r = strip_html_markdown(s)
        assert "<div>" not in r[0]
        assert "<a" not in r[0]
        assert "Fixed the" in r[0]
        assert "issue" in r[0]

    def test_markdown_link_extracted(self):
        from src.etl.cleaning import strip_html_markdown
        s = pd.Series(["See [docs](https://example.com) for details"])
        r = strip_html_markdown(s)
        assert "docs" in r[0]
        assert "https://" not in r[0]

    def test_plain_text_unchanged(self):
        from src.etl.cleaning import strip_html_markdown
        text = "Normal resolution text here"
        s = pd.Series([text])
        assert strip_html_markdown(s)[0] == text


# =============================================================================
# Unit tests — Safety Classifier
# =============================================================================

class TestSafetyClassifier:
    """Tests for governance.classifier.classify()"""

    def test_safe_select(self):
        from src.governance.classifier import classify, Classification
        r = classify("What is our current MRR?")
        assert r.classification == Classification.SAFE

    def test_safe_chart(self):
        from src.governance.classifier import classify, Classification
        r = classify("Generate a bar chart of monthly revenue")
        assert r.classification == Classification.SAFE

    def test_safe_policy(self):
        from src.governance.classifier import classify, Classification
        r = classify("What is our refund policy?")
        assert r.classification == Classification.SAFE

    def test_blocked_delete_question(self):
        from src.governance.classifier import classify, Classification
        r = classify("Delete all transactions from 2022")
        assert r.classification == Classification.BLOCKED

    def test_blocked_drop_question(self):
        from src.governance.classifier import classify, Classification
        r = classify("Drop the invoices table")
        assert r.classification == Classification.BLOCKED

    def test_blocked_sql_delete(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "DELETE FROM transactions WHERE amount < 0")
        assert r.classification == Classification.BLOCKED
        assert r.rule_triggered == "MUTATING_SQL"

    def test_blocked_sql_update(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "UPDATE subscriptions SET mrr=0")
        assert r.classification == Classification.BLOCKED

    def test_blocked_sql_drop(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "DROP TABLE customers")
        assert r.classification == Classification.BLOCKED

    def test_blocked_sql_insert(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "INSERT INTO customers VALUES (1,'a','b')")
        assert r.classification == Classification.BLOCKED

    def test_requires_review_pii_question(self):
        from src.governance.classifier import classify, Classification
        r = classify("Show me all customer email addresses")
        assert r.classification == Classification.REQUIRES_REVIEW

    def test_requires_review_pii_sql(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "SELECT email, phone FROM customers")
        assert r.classification == Classification.REQUIRES_REVIEW

    def test_requires_review_bulk_export(self):
        from src.governance.classifier import classify, Classification
        r = classify("Export all customer records")
        assert r.classification == Classification.REQUIRES_REVIEW

    def test_requires_review_financial_anomaly(self):
        from src.governance.classifier import classify, Classification
        r = classify("Are there any financial irregularities?")
        assert r.classification == Classification.REQUIRES_REVIEW

    def test_safe_sql_aggregate(self):
        from src.governance.classifier import classify, Classification
        r = classify("", "SELECT customer_id, SUM(amount) FROM transactions GROUP BY 1")
        assert r.classification == Classification.SAFE


# =============================================================================
# Unit tests — PII Masking
# =============================================================================

class TestPIIMasking:
    """Tests for governance.guardrails PII functions"""

    def test_email_local_masked(self):
        from src.governance.guardrails import mask_pii_in_text
        result = mask_pii_in_text("Contact john.doe@example.com for details")
        assert "john.doe@example.com" not in result
        assert "example.com" in result        # domain preserved
        assert "***" in result                # masking applied

    def test_no_pii_unchanged(self):
        from src.governance.guardrails import mask_pii_in_text
        text = "Revenue increased by 15% in Q3"
        assert mask_pii_in_text(text) == text

    def test_phone_masked(self):
        from src.governance.guardrails import mask_pii_in_text
        result = mask_pii_in_text("Call 555-867-5309 now")
        assert "867-5309" not in result

    def test_email_masked_in_dict(self):
        from src.governance.guardrails import mask_pii_in_value
        data = {"email": "jane@acme.com", "revenue": 5000}
        masked = mask_pii_in_value(data)
        assert "jane@acme.com" not in masked["email"]
        assert masked["revenue"] == 5000   # non-PII preserved

    def test_email_masked_in_list_of_dicts(self):
        from src.governance.guardrails import mask_pii_in_value
        data = [{"email": "a@b.com"}, {"email": "c@d.com"}]
        masked = mask_pii_in_value(data)
        assert all("a@b.com" not in r["email"] for r in masked)
        assert all("c@d.com" not in r["email"] for r in masked)

    def test_nested_masking(self):
        from src.governance.guardrails import mask_pii_in_value
        data = {"customer": {"email": "x@y.com", "id": "C1"}}
        masked = mask_pii_in_value(data)
        assert "x@y.com" not in masked["customer"]["email"]
        assert masked["customer"]["id"] == "C1"


class TestRowLimit:
    """Tests for guardrails.enforce_row_limit()"""

    def test_large_result_truncated(self):
        from src.governance.guardrails import enforce_row_limit
        big = [{"id": i} for i in range(1500)]
        result, warning = enforce_row_limit(big)
        assert len(result) == 1000
        assert warning is not None
        assert "1000" in warning

    def test_small_result_unchanged(self):
        from src.governance.guardrails import enforce_row_limit
        small = [{"id": i} for i in range(50)]
        result, warning = enforce_row_limit(small)
        assert len(result) == 50
        assert warning is None

    def test_non_list_unchanged(self):
        from src.governance.guardrails import enforce_row_limit
        data = {"key": "value"}
        result, warning = enforce_row_limit(data)
        assert result == data
        assert warning is None


class TestReflection:
    """Tests for guardrails.reflect()"""

    def test_good_answer_passes(self):
        from src.governance.guardrails import reflect
        r = reflect(
            question    = "What is our total revenue this quarter?",
            sql         = "SELECT SUM(amount) AS revenue FROM transactions WHERE status='completed'",
            answer      = "Total revenue this quarter is $1,234,567, an increase of 12%.",
            raw_results = [{"revenue": 1234567}],
        )
        assert r["passed"] is True
        assert len(r["notes"]) == 0

    def test_negative_amount_flagged(self):
        from src.governance.guardrails import reflect
        r = reflect(
            question    = "Show revenue by month",
            sql         = "SELECT month, SUM(amount) AS revenue FROM transactions GROUP BY month",
            answer      = "Monthly revenue data is shown below.",
            raw_results = [{"month": "2025-01", "revenue": -500}],
        )
        assert r["passed"] is False
        assert any("Negative" in note for note in r["notes"])

    def test_empty_answer_flagged(self):
        from src.governance.guardrails import reflect
        r = reflect(question="What is MRR?", sql=None, answer="", raw_results=None)
        assert r["passed"] is False


# =============================================================================
# Unit tests — Tools (mocked DataFrames)
# =============================================================================

class TestRevenueCalculatorTool:
    """Tests for tools.revenue_calculator_tool"""

    @pytest.fixture(autouse=True)
    def setup_dataframes(self):
        import src.agent.tools as T
        txn = pd.DataFrame({
            "transaction_id":   ["T1", "T2", "T3"],
            "customer_id":      ["C1", "C2", "C3"],
            "transaction_date": ["2025-01-15", "2025-02-15", "2025-03-15"],
            "amount":           [100.0, 200.0, 300.0],
            "status":           ["completed", "completed", "completed"],
        })
        subs = pd.DataFrame({
            "subscription_id": ["S1", "S2"],
            "customer_id":     ["C1", "C2"],
            "mrr":             [500.0, 300.0],
            "status":          ["active", "active"],
        })
        T._dataframes = {"transactions": txn, "subscriptions": subs}

    def test_returns_mrr(self):
        from src.agent.tools import _revenue_calculator
        r = _revenue_calculator("2025-01-01", "2025-12-31", "monthly", "all")
        assert r["mrr"] > 0  # real DB: ~197,397

    def test_returns_arr(self):
        from src.agent.tools import _revenue_calculator
        r = _revenue_calculator("2025-01-01", "2025-12-31", "monthly", "all")
        assert r["arr"] == pytest.approx(r["mrr"] * 12)

    def test_returns_arpc(self):
        from src.agent.tools import _revenue_calculator
        r = _revenue_calculator("2025-01-01", "2025-12-31", "monthly", "all")
        assert r["arpc"] > 0  # mrr / active customers

    def test_timeseries_returned(self):
        from src.agent.tools import _revenue_calculator
        r = _revenue_calculator("2025-01-01", "2025-12-31", "monthly", "all")
        assert isinstance(r["timeseries"], list)
        assert len(r["timeseries"]) >= 1

    def test_empty_period_no_crash(self):
        from src.agent.tools import _revenue_calculator
        r = _revenue_calculator("2010-01-01", "2010-12-31", "monthly", "all")
        assert isinstance(r, dict)


class TestAnomalyDetectionTool:
    """Tests for tools.anomaly_detection_tool"""

    @pytest.fixture(autouse=True)
    def setup_dataframes(self):
        import src.agent.tools as T
        from datetime import timedelta, date
        # 30 normal days + 1 spike — anchored to dataset max date
        ref = date(2025, 12, 31)
        dates = [(ref - timedelta(days=29-i)).strftime("%Y-%m-%d") for i in range(30)]
        amounts = [100.0] * 29 + [9999.0]   # last day is a spike
        txn = pd.DataFrame({
            "transaction_id":   [f"T{i}" for i in range(30)],
            "customer_id":      ["C1"] * 30,
            "transaction_date": dates,
            "amount":           amounts,
            "status":           ["completed"] * 30,
        })
        subs = pd.DataFrame(columns=["subscription_id","customer_id","mrr","status"])
        T._dataframes = {"transactions": txn, "subscriptions": subs}

    def test_detects_spike(self):
        from src.agent.tools import _anomaly_detection
        r = _anomaly_detection("revenue", 60, 2.0, "zscore")
        assert r["anomaly_count"] >= 1
        assert r["anomalies"][0]["direction"] == "spike"

    def test_returns_stats(self):
        from src.agent.tools import _anomaly_detection
        r = _anomaly_detection("revenue", 60, 2.0, "zscore")
        assert "mean" in r
        assert "data_points" in r
        assert r["data_points"] > 0

    def test_invalid_metric_returns_error(self):
        from src.agent.tools import _anomaly_detection
        r = _anomaly_detection("nonexistent_metric", 30, 2.0, "zscore")
        assert "error" in r


# =============================================================================
# Unit tests — Approval Queue
# =============================================================================

class TestApprovalQueue:
    """Tests for governance.queue"""

    def test_enqueue_returns_id(self):
        from src.governance.queue import enqueue
        qid = enqueue("test question", "requires_review", "PII detected")
        assert isinstance(qid, int)
        assert qid > 0

    def test_enqueued_appears_in_pending(self):
        from src.governance.queue import enqueue, list_pending
        qid = enqueue("pending test", "requires_review", "test reason")
        pending = list_pending()
        ids = [p["id"] for p in pending]
        assert qid in ids

    def test_approve_changes_status(self):
        from src.governance.queue import enqueue, approve, get_entry
        qid     = enqueue("approve test", "requires_review", "test")
        updated = approve(qid, reviewer_notes="Approved", reviewer_id="tester")
        assert updated["status"] == "approved"
        assert "tester" in updated["reviewer_notes"]

    def test_reject_changes_status(self):
        from src.governance.queue import enqueue, reject
        qid     = enqueue("reject test", "requires_review", "test")
        updated = reject(qid, reviewer_notes="No justification")
        assert updated["status"] == "rejected"

    def test_double_approve_raises(self):
        from src.governance.queue import enqueue, approve
        qid = enqueue("double approve test", "requires_review", "test")
        approve(qid)
        with pytest.raises(ValueError, match="already"):
            approve(qid)

    def test_unknown_id_raises(self):
        from src.governance.queue import approve
        with pytest.raises(ValueError, match="not found"):
            approve(999999)


# =============================================================================
# Unit tests — Audit Log
# =============================================================================

class TestAuditLog:
    """Tests for governance.audit"""

    def test_log_returns_id(self):
        from src.governance.audit import log_interaction
        lid = log_interaction(
            "test question", ["sql_query_tool"], "SELECT 1",
            "safe", "Test answer", 100.0,
        )
        assert isinstance(lid, int) and lid > 0

    def test_logged_entry_retrievable(self):
        from src.governance.audit import log_interaction, get_history
        question = f"unique_test_q_{id(object())}"
        lid = log_interaction(question, ["revenue_calculator_tool"], None,
                              "safe", "answer", 200.0)
        _, rows = get_history(limit=100)
        questions = [r["user_question"] for r in rows]
        assert question in questions

    def test_filter_by_classification(self):
        from src.governance.audit import log_interaction, get_history
        log_interaction("blocked q", [], None, "blocked", "blocked", 5.0,
                        approval_status="blocked")
        total, rows = get_history(limit=50, class_filter="blocked")
        assert all(r["classification"] == "blocked" for r in rows)

    def test_filter_by_tool(self):
        from src.governance.audit import log_interaction, get_history
        tool_name = "chart_generator_tool"
        log_interaction("chart question", [tool_name], None, "safe", "chart", 300.0)
        total, rows = get_history(limit=50, tool_filter=tool_name)
        assert len(rows) >= 1
        assert all(tool_name in r["tools_used"] for r in rows)


# =============================================================================
# Integration tests — FastAPI endpoints
# =============================================================================

class TestFastAPIEndpoints:
    """FastAPI integration tests via TestClient"""

    @pytest.fixture(autouse=True)
    def client(self):
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
        from fastapi.testclient import TestClient
        from src.api.main import app
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_health_returns_200(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "db_status" in data
        assert "row_counts" in data
        assert "agent_ready" in data

    def test_health_db_connected(self):
        r = self.client.get("/health")
        if r.status_code == 200:
            assert r.json()["db_status"] == "connected"

    def test_etl_run_returns_202(self):
        r = self.client.post("/etl/run")
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "accepted"

    def test_etl_status_404_unknown_job(self):
        r = self.client.get("/etl/status/definitely-not-a-real-job-id")
        assert r.status_code == 404

    def test_validate_sql_safe(self):
        r = self.client.post("/agent/query/validate",
                             json={"sql": "SELECT COUNT(*) FROM customers"})
        assert r.status_code == 200
        assert r.json()["classification"] == "safe"

    def test_validate_sql_blocked_delete(self):
        r = self.client.post("/agent/query/validate",
                             json={"sql": "DELETE FROM transactions"})
        assert r.status_code == 200
        assert r.json()["classification"] == "blocked"

    def test_validate_sql_blocked_update(self):
        r = self.client.post("/agent/query/validate",
                             json={"sql": "UPDATE subscriptions SET mrr=0"})
        assert r.status_code == 200
        assert r.json()["classification"] == "blocked"

    def test_validate_sql_requires_review_pii(self):
        r = self.client.post("/agent/query/validate",
                             json={"sql": "SELECT email, phone FROM customers"})
        assert r.status_code == 200
        assert r.json()["classification"] == "requires_review"

    def test_analytics_dashboard(self):
        r = self.client.get("/analytics/dashboard")
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            data = r.json()
            assert "mrr" in data
            assert "arr" in data
            assert data["arr"] == pytest.approx(data["mrr"] * 12, rel=0.01)

    def test_governance_pending_returns_list(self):
        r = self.client.get("/governance/pending")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_governance_review_404_unknown_id(self):
        r = self.client.post("/governance/review/999999",
                             json={"action": "approve", "reviewer_notes": "test"})
        assert r.status_code == 404

    def test_query_history_returns_paginated(self):
        r = self.client.get("/agent/query/history?limit=5&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "entries" in data
        assert "limit" in data
        assert data["limit"] == 5

    def test_query_history_tool_filter(self):
        r = self.client.get("/agent/query/history?tool_used=sql_query_tool")
        assert r.status_code == 200

    def test_query_history_class_filter(self):
        r = self.client.get("/agent/query/history?classification=safe")
        assert r.status_code == 200
        data = r.json()
        for entry in data["entries"]:
            assert entry["classification"] == "safe"