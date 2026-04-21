"""
src/agent/agent.py
==================
Step 4 — FinOps LangGraph Agent  (LangChain 1.x compatible)

LangChain 1.x removed AgentExecutor. This file uses the current
recommended pattern:

    langchain.agents.create_agent
        + MemorySaver   (conversation memory across turns)
        + SystemMessage (tool routing instructions)

Architecture
------------
  LLM         : Claude claude-opus-4-5 via langchain-anthropic
  Framework   : LangChain create_agent (LangGraph-backed; replaces AgentExecutor in 1.x)
  Memory      : MemorySaver + thread_id (persistent across turns per session)
  Tools       : 7 tools from tools.py
  Routing     : Explicit rules in system prompt
  Recovery    : max_iterations via recursion_limit on the graph config
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]

# =============================================================================
# System prompt — tool routing rules, date anchoring, output format
# =============================================================================

SYSTEM_PROMPT = """You are a FinOps Analytics AI assistant for a B2B SaaS company.
You have access to a SQLite database (finops.db) and a set of specialised tools.
Always use the most appropriate tool. Never answer data or policy questions from memory.

TOOL ROUTING RULES (follow strictly)
--------------------------------------
1. POLICY / SLA / REFUND / PRICING / ESCALATION questions
   → ALWAYS use knowledge_retrieval_tool
   → Never invent policy details — only use retrieved text

2. REVENUE METRICS  (MRR, ARR, ARPC, growth rate)
   → ALWAYS use revenue_calculator_tool
   → Do NOT write SQL for revenue computations

3. CUSTOMER SEGMENTATION  (RFM, churn risk, usage tiers)
   → ALWAYS use customer_segmentation_tool

4. ANOMALY DETECTION  (unusual patterns, spikes, drops)
   → ALWAYS use anomaly_detection_tool

5. CHART / VISUALISATION requests
   → ALWAYS use chart_generator_tool
   → Provide a valid SQL query in the data_sql parameter

6. FORECASTING / PREDICTION / FUTURE TREND requests
   → ALWAYS use forecast_tool 

7. ALL OTHER DATA QUESTIONS  (lookups, counts, joins, cohort data)
   → use sql_query_tool

MULTI-STEP QUESTIONS
---------------------
If a question needs multiple tools, call them in sequence before answering.
Examples:
  "Resolution time for high-priority tickets AND what does the SLA say?"
  → Step 1: sql_query_tool  → Step 2: knowledge_retrieval_tool

  "Compare churn rates AND forecast the trend"
  → Step 1: sql_query_tool  → Step 2: forecast_tool

DATASET NOTES
-------------
- All data ends 2025-12-31. For "last N days" use: date('2025-12-31', '-N days')
- plan_name values: 'basic', 'pro', 'scale', 'growth', 'enterprise', 'free'
- Completed revenue: WHERE status = 'completed'
- Active subscriptions: WHERE status = 'active'

OUTPUT FORMAT
-------------
- Lead with the key number or finding.
- For policy answers, always cite: [Source: filename.md — Section Name]
- For data answers, include the actual numbers.
- If a chart was saved, say: Chart saved to: <path>
"""


# =============================================================================
# Agent class
# =============================================================================

class FinOpsAgent:
    """
    LangGraph-based multi-tool FinOps agent.

    Uses create_agent (LangChain agents API) with
    MemorySaver for per-session conversation history.

    Each call to .query() uses a stable thread_id so LangGraph's
    MemorySaver replays history automatically on subsequent turns.
    """

    def __init__(self, verbose: bool = False):
        self.verbose   = verbose
        self._ready    = False
        self.thread_id = str(uuid.uuid4())   # unique session id
        self._build()

    def _build(self) -> None:
        from langchain_anthropic import ChatAnthropic
        from langchain.agents import create_agent
        from langchain_core.messages import SystemMessage
        from langgraph.checkpoint.memory import MemorySaver
        from src.agent.tools import ALL_TOOLS

        self.llm = ChatAnthropic(
            model="claude-opus-4-5",
            temperature=0,
            max_tokens=2048,
        )

        self.tools     = ALL_TOOLS
        self.tools_map = {t.name: t for t in self.tools}
        self.memory    = MemorySaver()

        # create_agent: system_prompt accepts str or SystemMessage
        self.graph = create_agent(
            model          = self.llm,
            tools          = self.tools,
            system_prompt  = SystemMessage(content=SYSTEM_PROMPT),
            checkpointer   = self.memory,
        )

        self._ready = True
        log.info("FinOpsAgent ready — %d tools", len(self.tools))

    def query(self, question: str, session_id: Optional[str] = None) -> dict[str, Any]:
        """
        Run a natural language question through the agent.

        Parameters
        ----------
        question   : the user's question
        session_id : optional override; if None uses self.thread_id
                     (all calls share memory within the same FinOpsAgent instance)

        Returns
        -------
        dict:
            answer            str
            tools_used        list[str]
            generated_sql     str | None
            raw_results       list | None
            sources           list | None
            chart_path        str | None
            execution_time_ms float
            confidence        float
        """
        if not self._ready:
            return {"error": "Agent not initialised"}

        from langchain_core.messages import HumanMessage

        thread_id   = session_id or self.thread_id
        config      = {"configurable": {"thread_id": thread_id},
                       "recursion_limit": 25}

        t0 = time.perf_counter()
        tools_used:  list[str]       = []
        sql:         Optional[str]   = None
        raw:         Any             = None
        sources:     Any             = None
        chart_path:  Optional[str]   = None

        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )

            # ── Extract final answer ─────────────────────────────────────────
            messages = result.get("messages", [])
            answer   = ""
            for msg in reversed(messages):
                # Last AIMessage that isn't a tool call = final answer
                if hasattr(msg, "content") and msg.content and \
                   not getattr(msg, "tool_calls", None):
                    if hasattr(msg, "type") and msg.type == "ai":
                        answer = msg.content if isinstance(msg.content, str) \
                                 else str(msg.content)
                        break
                    elif msg.__class__.__name__ == "AIMessage":
                        answer = msg.content if isinstance(msg.content, str) \
                                 else str(msg.content)
                        break

            if not answer:
                answer = str(messages[-1].content) if messages else "No response generated"

            # ── Parse tool use from message history ──────────────────────────
            for msg in messages:
                # ToolMessage carries the tool name
                if msg.__class__.__name__ == "ToolMessage":
                    tname = getattr(msg, "name", "")
                    if tname and tname not in tools_used:
                        tools_used.append(tname)
                    # Parse observation content
                    content = getattr(msg, "content", "")
                    if isinstance(content, str):
                        import json as _json
                        try:
                            obs = _json.loads(content)
                        except Exception:
                            obs = {}
                    elif isinstance(content, dict):
                        obs = content
                    else:
                        obs = {}

                    if tname == "sql_query_tool":
                        sql = obs.get("sql") or sql
                        raw = obs.get("data") or raw

                    if tname == "knowledge_retrieval_tool":
                        passages = obs.get("passages", [])
                        if passages:
                            sources = [
                                {"source": p.get("source",""),
                                 "section": p.get("section","")}
                                for p in passages
                            ]

                    if tname == "chart_generator_tool":
                        chart_path = obs.get("file_path") or chart_path

            # ── Confidence heuristic ─────────────────────────────────────────
            if not tools_used:
                confidence = 0.50
            elif "error" in answer.lower()[:80]:
                confidence = 0.30
            else:
                confidence = 0.90

            if self.verbose:
                print(f"\n[DEBUG] tools_used={tools_used}, answer_len={len(answer)}")

        except Exception as e:
            log.error("Agent execution error: %s", e)
            answer     = f"I encountered an error processing your question: {e}"
            confidence = 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer":            answer,
            "tools_used":        tools_used,
            "generated_sql":     sql,
            "raw_results":       raw,
            "sources":           sources,
            "chart_path":        chart_path,
            "execution_time_ms": round(elapsed_ms, 1),
            "confidence":        confidence,
        }

    def reset_memory(self) -> None:
        """Start a new conversation thread (fresh memory)."""
        self.thread_id = str(uuid.uuid4())
        log.info("Conversation memory reset — new thread_id: %s", self.thread_id)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def tool_names(self) -> list[str]:
        return list(self.tools_map.keys())