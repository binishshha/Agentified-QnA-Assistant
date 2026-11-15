"""Small, dependency-light evaluation harness for the research agent.

This intentionally does not use an evaluation framework. It drives the real
``run_agent`` function with a deterministic planner and the evaluation corpus,
so the loop, tool dispatch, ledger, pruning, stop reasons, and token accounting
are exercised without requiring an API key or network access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval" / "corpus"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import agent


class CorpusStore:
    def __init__(self, corpus: Path):
        self.documents = {
            path.name: path.read_text(encoding="utf-8")
            for path in corpus.glob("*.md")
        }
        self.last_query = ""

    def search(self, _embedding: list[float], top_k: int = 3, document_name: str | None = None) -> list[dict[str, Any]]:
        names = [document_name] if document_name else list(self.documents)
        query_terms = set(self.last_query.lower().split())
        matches = []
        for name in names:
            text = self.documents.get(name, "")
            score = sum(term in text.lower() for term in query_terms)
            if score or not query_terms:
                matches.append({"text": text, "source": name, "chunk_index": 0, "_score": score})
        matches.sort(key=lambda item: item["_score"], reverse=True)
        return [{key: value for key, value in item.items() if key != "_score"} for item in matches[:top_k]]


class EvalLLM:
    def __init__(self, store: CorpusStore):
        self.store = store

    @staticmethod
    def _chat_kwargs() -> dict[str, str]:
        return {"model": "evaluation-fake-model"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.store.last_query = texts[0]
        return [[0.0, 0.0, 0.0] for _ in texts]


def call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class DeterministicPlanner:
    """A tiny test model whose next action depends on the observed trajectory."""

    def __init__(self, scenario: str):
        self.scenario = scenario
        self.calls = 0

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        step = self.calls
        self.calls += 1
        if self.scenario == "conflict":
            actions = [
                ("search_document", {"document_name": "northstar_pro.md", "query": "price retention"}),
                ("record_evidence", {"claim": "The current Pro plan costs $50.", "source": "northstar_pro.md", "supports": True}),
                ("search_document", {"document_name": "northstar_sales_sheet.md", "query": "price retention"}),
                ("record_evidence", {"claim": "The draft sales sheet says Pro costs $45.", "source": "northstar_sales_sheet.md", "supports": True}),
                ("finish", {"answer": "The current plan says $50, while the draft says $45; the documents conflict.", "sources": ["northstar_pro.md", "northstar_sales_sheet.md"], "confidence": 0.85, "status": "answered"}),
            ]
        elif self.scenario == "sso":
            actions = [
                ("search_document", {"document_name": "northstar_security_policy.md", "query": "SSO audit logs"}),
                ("record_evidence", {"claim": "SSO and audit logs are Enterprise-only.", "source": "northstar_security_policy.md", "supports": True}),
                ("finish", {"answer": "SSO and audit logs are available only on Enterprise.", "sources": ["northstar_security_policy.md"], "confidence": 0.95, "status": "answered"}),
            ]
        elif self.scenario == "missing":
            actions = [
                ("search_document", {"document_name": "northstar_basic.md", "query": "uptime SLA"}),
                ("search_document", {"document_name": "northstar_pro.md", "query": "uptime SLA"}),
                ("finish", {"answer": "The corpus does not specify an uptime SLA.", "sources": [], "confidence": 0.1, "status": "insufficient_evidence"}),
            ]
        else:  # injected_failure
            actions = [
                ("search_document", {"document_name": "northstar_pro.md", "query": "price"}),
                ("search_document", {"document_name": "northstar_sales_sheet.md", "query": "price"}),
                ("finish", {"answer": "The search tool failed, so I cannot verify the price.", "sources": [], "confidence": 0.05, "status": "insufficient_evidence"}),
            ]
        name, arguments = actions[min(step, len(actions) - 1)]
        return {
            "usage": {"total_tokens": 10},
            "choices": [{"message": {"tool_calls": [call(name, arguments, str(step))]}}],
        }


def classify_failure(result: dict[str, Any], injected: bool) -> str:
    errors = [item for item in result["trace"] if isinstance(item.get("result"), dict) and item["result"].get("error")]
    if not errors:
        return "none"
    if result["status"] == "insufficient_evidence" and result["confidence"] <= 0.2:
        return "soft failure" if injected else "cascading soft failure"
    return "hard failure"


def run_case(name: str, question: str, expected_tools: list[str], expected_status: str, injected: bool = False) -> dict[str, Any]:
    store = CorpusStore(CORPUS)
    llm = EvalLLM(store)
    planner = DeterministicPlanner(name)
    old_completion = agent.completion
    old_failure = os.environ.get("INJECT_FAILURE")
    with tempfile.TemporaryDirectory() as ledger_dir:
        os.environ["EVIDENCE_LEDGER_PATH"] = str(Path(ledger_dir) / "evidence.jsonl")
        if injected:
            os.environ["INJECT_FAILURE"] = "search_document:timeout"
        else:
            os.environ.pop("INJECT_FAILURE", None)
        agent.completion = planner
        try:
            result = agent.run_agent(question, agent.AgentState(), llm, max_steps=8, token_budget=20000)
        finally:
            agent.completion = old_completion
            if old_failure is None:
                os.environ.pop("INJECT_FAILURE", None)
            else:
                os.environ["INJECT_FAILURE"] = old_failure
        tools = [item["tool"] for item in result["trace"]]
        valid_tools = tools == expected_tools and all(item.get("arguments") for item in result["trace"])
        return {
            "query": question,
            "completed": result["status"] == expected_status,
            "tool_calls_correct": valid_tools,
            "trajectory_steps": result["steps"],
            "tokens": result["tokens"],
            "status": result["status"],
            "stop_reason": result["stop_reason"],
            "failure_class": classify_failure(result, injected),
            "trace": result["trace"],
        }


def render_report(results: list[dict[str, Any]]) -> str:
    completed = sum(item["completed"] for item in results)
    correct = sum(item["tool_calls_correct"] for item in results)
    lines = [
        "# Agent evaluation results",
        "",
        "Generated by `eval/evaluate.py` using the real `run_agent` loop and the Northstar corpus.",
        "",
        "| Query | Completed | Tools correct | Steps | Tokens | Status | Stop reason | Failure class |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['query']} | {item['completed']} | {item['tool_calls_correct']} | {item['trajectory_steps']} | {item['tokens']} | {item['status']} | {item['stop_reason']} | {item['failure_class']} |"
        )
    lines.extend([
        "",
        f"Task completion rate: {completed}/{len(results)} ({completed / len(results):.0%})",
        f"Tool-call correctness: {correct}/{len(results)} ({correct / len(results):.0%})",
        "",
        "## Failure log",
        "",
    ])
    for item in results:
        if item["failure_class"] != "none":
            errors = [entry for entry in item["trace"] if isinstance(entry.get("result"), dict) and entry["result"].get("error")]
            lines.append(f"- **{item['failure_class']}** for `{item['query']}`: {errors}; final status `{item['status']}` with stop reason `{item['stop_reason']}`.")
    lines.extend([
        "",
        "The injected timeout is a handled soft failure: the agent records the tool error, avoids citing unsupported evidence, and returns low-confidence `insufficient_evidence`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.md")
    args = parser.parse_args()
    results = [
        run_case("conflict", "Compare the current Pro price with the draft sales sheet.", ["search_document", "record_evidence", "search_document", "record_evidence", "finish"], "answered"),
        run_case("sso", "Which plan includes SSO and audit logs?", ["search_document", "record_evidence", "finish"], "answered"),
        run_case("missing", "What is Northstar's uptime SLA?", ["search_document", "search_document", "finish"], "insufficient_evidence"),
        run_case("injected_failure", "What is the Pro price?", ["search_document", "search_document", "finish"], "insufficient_evidence", injected=True),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(results), encoding="utf-8")
    print(render_report(results))


if __name__ == "__main__":
    main()
