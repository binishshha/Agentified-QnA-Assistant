import ast
import json
import os
import operator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .vector_store import VectorStore

EVIDENCE_LEDGER_PATH = Path("data/evidence_ledger.jsonl")
FINISH_STATUSES = {"answered", "insufficient_evidence", "needs_clarification"}


def _ledger_path() -> Path:
    """Return the evidence-ledger path, allowing tests and deployments to override it."""
    return Path(os.environ.get("EVIDENCE_LEDGER_PATH", str(EVIDENCE_LEDGER_PATH)))


def _record_evidence(claim: str, source: str, supports: bool) -> dict[str, Any]:
    if not claim.strip():
        raise ValueError("claim must not be blank")
    if not source.strip():
        raise ValueError("source must not be blank")

    entry = {
        "claim": claim,
        "source": source,
        "supports": supports,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _ask_user(question: str) -> dict[str, str]:
    if not question.strip():
        raise ValueError("question must not be blank")
    answer = input(f"{question}\n> ").strip()
    return {"question": question, "answer": answer}


def _finish(
    answer: str,
    sources: list[str],
    confidence: float,
    status: str,
) -> dict[str, Any]:
    if status not in FINISH_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(FINISH_STATUSES))}")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not answer.strip():
        raise ValueError("answer must not be blank")
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "status": status,
    }


def _calculate(expression: str) -> float | int:
    operators: dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    def visit(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Only basic arithmetic is allowed")

    tree = ast.parse(expression, mode="eval")
    result = visit(tree)
    if abs(result) > 1e100:
        raise ValueError("Result is too large")
    return result


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_document_metadata",
                "description": "Get document metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "field": {"type": "string"},
                    },
                    "required": ["document_name", "field"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "List ingested documents",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": "Search one document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["document_name", "query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate basic arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_evidence",
                "description": "Record whether a source supports a claim in the evidence ledger",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source": {"type": "string"},
                        "supports": {"type": "boolean"},
                    },
                    "required": ["claim", "source", "supports"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Pause and ask the user a clarification question in the terminal",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Return a grounded final answer with sources, confidence, and status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "status": {
                            "type": "string",
                            "enum": [
                                "answered",
                                "insufficient_evidence",
                                "needs_clarification",
                            ],
                        },
                    },
                    "required": ["answer", "sources", "confidence", "status"],
                },
            },
        },
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    store: VectorStore,
    embed: Callable[[list[str]], list[list[float]]],
) -> Any:
    if name == "lookup_document_metadata":
        metadata = store.document_metadata(arguments["document_name"])
        field = arguments["field"]
        if field not in metadata:
            raise ValueError(f"Unknown metadata field: {field}")
        return {field: metadata[field]}
    if name == "list_documents":
        return store.list_documents()
    if name == "calculate":
        return {"result": _calculate(arguments["expression"])}
    if name == "search_document":
        results = store.search(
            embed([arguments["query"]])[0],
            top_k=3,
            document_name=arguments["document_name"],
        )
        return [{**result, "text": result["text"][:400]} for result in results]
    if name == "record_evidence":
        return _record_evidence(
            arguments["claim"], arguments["source"], arguments["supports"]
        )
    if name == "ask_user":
        return _ask_user(arguments["question"])
    if name == "finish":
        return _finish(
            arguments["answer"],
            arguments["sources"],
            arguments["confidence"],
            arguments["status"],
        )
    raise ValueError(f"Unknown tool: {name}")
