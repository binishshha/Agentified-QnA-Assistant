"""Single-agent research loop with bounded context and explicit stop reasons."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from litellm import completion
except ImportError:  # pragma: no cover - the application dependency provides this.
    completion = None

from .tools import execute_tool, tool_definitions


AGENT_PROMPT = """You are a careful document research agent.

Use the supplied tools to investigate the user's question. Search documents,
record every useful claim with record_evidence, and do not treat a draft or a
single conflicting source as settled fact. If the question is ambiguous, use
ask_user. Always finish with the finish tool. Use status answered only when the
ledger supports the answer; use insufficient_evidence when the documents do
not establish it, and needs_clarification when the user must choose.

Every cited source must be a source you recorded in the evidence ledger.
"""


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _call_name(call: Any) -> str:
    function = _value(call, "function", {})
    return _value(function, "name", "") or _value(call, "name", "")


def _call_arguments(call: Any) -> dict[str, Any]:
    function = _value(call, "function", {})
    arguments = _value(function, "arguments", None)
    if arguments is None:
        arguments = _value(call, "arguments", {})
    if isinstance(arguments, str):
        return json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    return arguments


def _call_message(call: Any) -> dict[str, Any]:
    function = _value(call, "function", {})
    return {
        "id": _value(call, "id", ""),
        "type": "function",
        "function": {
            "name": _call_name(call),
            "arguments": _value(function, "arguments", "{}"),
        },
    }


def _ledger_path() -> str:
    return os.environ.get("EVIDENCE_LEDGER_PATH", "data/evidence_ledger.jsonl")


def _ledger_line_count() -> int:
    """Lines already in the ledger, so a run can ignore other runs' entries."""
    try:
        with open(_ledger_path(), encoding="utf-8") as ledger:
            return sum(1 for line in ledger if line.strip())
    except OSError:
        return 0


def _ledger_lines(start: int = 0) -> list[str]:
    try:
        with open(_ledger_path(), encoding="utf-8") as ledger:
            return [line.strip() for line in ledger if line.strip()][start:]
    except OSError:
        return []


def _ledger_sources(start: int = 0) -> set[str]:
    sources: set[str] = set()
    for line in _ledger_lines(start):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("source"):
            sources.add(str(item["source"]))
    return sources


def _sources_are_recorded(sources: list[str], start: int = 0) -> bool:
    recorded = _ledger_sources(start)
    return all(
        any(source == item or source.startswith(f"{item} chunk ") for item in recorded)
        for source in sources
    )


def _calls_from_content(content: str, step: int) -> list[dict[str, Any]]:
    """Recover tool calls a small model wrote as JSON text instead of tool_calls.

    Accepts {"name": ..., "arguments"|"parameters": {...}}, the OpenAI-style
    {"function": {...}} wrapper, a list of those, optionally inside ``` fences.
    """
    known = {tool["function"]["name"] for tool in tool_definitions()}
    text = (content or "").replace("```json", "```").replace("```", " ")
    decoder = json.JSONDecoder()
    found: list[Any] = []
    index = 0
    while index < len(text):
        if text[index] in "{[":
            try:
                value, end = decoder.raw_decode(text, index)
                found.extend(value if isinstance(value, list) else [value])
                index = end
                continue
            except json.JSONDecodeError:
                pass
        index += 1
    calls: list[dict[str, Any]] = []
    for item in found:
        if not isinstance(item, dict):
            continue
        item = item.get("function") if isinstance(item.get("function"), dict) else item
        name = item.get("name")
        arguments = item.get("arguments", item.get("parameters", {}))
        if name not in known:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                continue
        if not isinstance(arguments, dict):
            continue
        calls.append({
            "id": f"text_call_{step}_{len(calls)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        })
    return calls


def cap(value: Any, limit: int = 4000) -> Any:
    """Keep tool output small enough that one result cannot dominate context."""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [cap(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: cap(item, limit) for key, item in value.items()}
    return value


def _ledger_block(start: int = 0) -> str:
    lines = _ledger_lines(start)[-12:]
    return "Current evidence ledger (this question only):\n" + ("\n".join(lines) or "(empty)")


def prune(messages: list[dict[str, Any]], state: "AgentState") -> list[dict[str, Any]]:
    """Clear old tool payloads while injecting the durable evidence ledger."""
    result: list[dict[str, Any]] = []
    tool_indexes = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    keep = set(tool_indexes[-2:])
    for index, message in enumerate(messages):
        copied = dict(message)
        if copied.get("role") == "tool" and index not in keep:
            copied["content"] = "[cleared – see notes]"
        result.append(copied)
    result = [message for message in result if message.get("role") != "system" or not message.get("content", "").startswith("Current evidence ledger")]
    result.append({"role": "system", "content": _ledger_block(state.ledger_start)})
    return result


@dataclass
class AgentState:
    tokens: int = 0
    steps: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    nudges: int = 0
    ledger_start: int = 0

    def add_usage(self, usage: Any) -> None:
        total = _value(usage, "total_tokens", None)
        if total is None:
            total = (_value(usage, "prompt_tokens", 0) or 0) + (_value(usage, "completion_tokens", 0) or 0)
        self.tokens += int(total or 0)

    def append(self, step: int, call: Any, result: Any) -> None:
        self.trace.append({
            "step": step,
            "tool": _call_name(call),
            "arguments": _call_arguments(call),
            "result": cap(result),
        })

    def finalize(self, call: Any) -> dict[str, Any]:
        arguments = _call_arguments(call)
        name = _call_name(call)
        if name == "ask_user":
            return {
                "answer": arguments.get("question", "Clarification is required."),
                "status": "needs_clarification",
                "sources": [],
                "confidence": 0.0,
                "stop_reason": "ask_user",
            }
        answer = str(arguments.get("answer", ""))
        sources = list(arguments.get("sources", []))
        confidence = float(arguments.get("confidence", 0.0))
        status = str(arguments.get("status", "insufficient_evidence"))
        if sources and not _sources_are_recorded(sources, self.ledger_start):
            status = "insufficient_evidence"
            confidence = min(confidence, 0.2)
            answer += " Evidence citations were not fully recorded in the ledger."
        return {
            "answer": answer,
            "status": status,
            "sources": sources,
            "confidence": max(0.0, min(confidence, 1.0)),
            "stop_reason": "finish",
        }

    def stop(self, reason: str) -> dict[str, Any]:
        return {
            "answer": "The research run did not produce a complete answer.",
            "status": "insufficient_evidence",
            "sources": [],
            "confidence": 0.0,
            "stop_reason": reason,
        }


def _completion_kwargs(llm: Any) -> dict[str, Any]:
    if hasattr(llm, "_chat_kwargs"):
        return llm._chat_kwargs()
    primary = getattr(llm, "primary", None)
    if primary is not None and hasattr(primary, "_chat_kwargs"):
        return primary._chat_kwargs()
    raise ValueError("llm must provide _chat_kwargs()")


def _completion_providers(llm: Any) -> list[Any]:
    primary = getattr(llm, "primary", None)
    if primary is None:
        return [llm]
    fallback = getattr(llm, "fallback", None)
    return [primary] + ([fallback] if fallback is not None else [])


def safe_execute(
    call: Any,
    store: Any,
    embed: Callable[[list[str]], list[list[float]]] | None,
) -> Any:
    """Execute a tool without raising, including deterministic failure injection."""
    name = _call_name(call)
    injected = os.environ.get("INJECT_FAILURE", "")
    failure_name, _, failure_mode = injected.partition(":")
    if failure_name == name:
        if failure_mode == "timeout":
            return {"error": "Injected timeout while executing tool"}
        if failure_mode == "malformed":
            return {"malformed": True, "error": "Injected malformed tool result"}
    try:
        return cap(execute_tool(name, _call_arguments(call), store, embed))
    except Exception as error:
        return {"error": str(error)}


def _retrieved_passages(state: AgentState) -> list[dict[str, Any]]:
    seen, passages = set(), []
    for entry in state.trace:
        if entry.get("tool") != "search_document" or not isinstance(entry.get("result"), list):
            continue
        for item in entry["result"]:
            if isinstance(item, dict) and item.get("text") and item.get("source"):
                key = (item["source"], item.get("chunk_index"))
                if key not in seen:
                    seen.add(key)
                    passages.append(item)
    return passages


def _synthesize_from_retrieved(question: str, state: AgentState, providers: list[Any]) -> dict[str, Any] | None:
    """Last resort for models that cannot drive the tool loop (small local models).

    One plain JSON completion, no tools, over the passages this run actually
    retrieved. Cited sources must be ones that were really retrieved, and
    confidence is capped because the model did not record its own evidence.
    """
    passages = _retrieved_passages(state)
    if not passages or completion is None:
        return None
    context = "\n\n".join(f"[{p['source']} chunk {p.get('chunk_index', 0)}] {p['text']}" for p in passages)
    messages = [
        {"role": "system", "content": (
            "Answer ONLY from the passages. If they do not contain the answer, say so. "
            'Return ONLY JSON: {"answer": string, "sources": [file names], "confidence": 0-1, '
            '"status": "answered" or "insufficient_evidence"}.')},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]
    for provider in providers:
        try:
            kwargs = _completion_kwargs(provider)
            kwargs.update({"messages": messages, "temperature": 0, "response_format": {"type": "json_object"}})
            response = completion(**kwargs)
            state.add_usage(_value(response, "usage", {}))
            message = _value((_value(response, "choices", []) or [{}])[0], "message", {})
            data = json.loads(_value(message, "content", "") or "{}")
            answer = str(data.get("answer", "")).strip()
            if not answer:
                continue
            retrieved = {p["source"] for p in passages}
            cited = [str(x).split(" chunk ")[0] for x in data.get("sources", [])]
            cited = [x for x in dict.fromkeys(cited) if x in retrieved] or sorted(retrieved)
            status = "answered" if data.get("status") == "answered" else "insufficient_evidence"
            state.trace.append({"step": state.steps, "tool": "synthesized_fallback",
                                "result": {"note": "answer written from retrieved passages without tool calls"}})
            return {
                "answer": answer + " (Answered from retrieved passages; the model did not record evidence.)",
                "status": status,
                "sources": cited,
                "confidence": max(0.0, min(float(data.get("confidence", 0.5)), 0.6)),
                "stop_reason": "synthesized_fallback",
            }
        except Exception as error:
            state.trace.append({"step": state.steps, "tool": "synthesized_fallback_error", "result": {"error": str(error)}})
    return None


def _finish_result(state: AgentState, result: dict[str, Any], reason: str | None = None) -> dict[str, Any]:
    result = dict(result)
    if reason:
        result["stop_reason"] = reason
    result.update({"steps": state.steps, "tokens": state.tokens, "trace": state.trace})
    return result


def run_agent(
    question: str,
    state: AgentState,
    llm: Any,
    max_steps: int = 8,
    token_budget: int = 20000,
) -> dict[str, Any]:
    state.ledger_start = _ledger_line_count()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": question},
    ]
    providers = _completion_providers(llm)
    store = getattr(llm, "store", None) or getattr(providers[0], "store", None)
    embed = getattr(llm, "embed", None) or getattr(providers[0], "embed", None)
    if completion is None:
        return _finish_result(state, state.stop("completion_unavailable"))

    for step in range(max_steps):
        state.steps = step + 1
        force_finish = step == max_steps - 1
        response = None
        completion_errors: list[str] = []
        for provider_index, provider in enumerate(providers):
            kwargs = _completion_kwargs(provider)
            kwargs.update({
                "messages": prune(messages, state),
                "tools": tool_definitions(),
                "tool_choice": (
                    {"type": "function", "function": {"name": "finish"}}
                    if force_finish
                    else "auto"
                ),
            })
            try:
                response = completion(**kwargs)
                if provider_index:
                    state.trace.append({
                        "step": step + 1,
                        "tool": "completion_fallback",
                        "result": {"provider": provider.settings.llm_provider},
                    })
                break
            except Exception as error:
                completion_errors.append(str(error))
                state.trace.append({
                    "step": step + 1,
                    "tool": "completion",
                    "result": {"provider": getattr(provider.settings, "llm_provider", "unknown"), "error": str(error)},
                })
        if response is None:
            return _finish_result(state, state.stop("completion_error"))
        state.add_usage(_value(response, "usage", {}))
        choices = _value(response, "choices", []) or []
        message = _value(choices[0], "message", {}) if choices else {}
        content = _value(message, "content", "") or ""
        calls = _value(message, "tool_calls", None) or []
        if not calls and content:
            calls = _calls_from_content(content, step + 1)
            if calls:
                state.trace.append({"step": step + 1, "tool": "text_tool_call_recovered",
                                    "result": {"note": "model wrote the tool call as text"}})
        if not calls:
            state.trace.append({"step": step + 1, "tool": "no_tool_call",
                                "result": {"model_said": str(content)[:500] or "(empty reply)"}})
            if state.nudges < 2:
                state.nudges += 1
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": (
                        "You must call a tool. Reply with a tool call only, for example "
                        '{"name": "search_document", "arguments": {"document_name": "<file>", "query": "<terms>"}}. '
                        "Call list_documents first if you do not know the file names. "
                        "When done, call finish."
                    )},
                ])
                continue
            fallback = _synthesize_from_retrieved(question, state, providers)
            return _finish_result(state, fallback or state.stop("no_tool_call"))

        messages.append({
            "role": "assistant",
            "content": _value(message, "content", "") or "",
            "tool_calls": [_call_message(call) for call in calls],
        })
        for call in calls:
            name = _call_name(call)
            if name in {"finish", "ask_user"}:
                result = state.finalize(call)
                state.append(step + 1, call, result)
                return _finish_result(state, result)
            result = safe_execute(call, store, embed)
            state.append(step + 1, call, result)
            messages.append({
                "role": "tool",
                "tool_call_id": _value(call, "id", ""),
                "content": json.dumps(cap(result), ensure_ascii=False),
            })
        if state.tokens > token_budget:
            return _finish_result(state, state.stop("budget_exceeded"))
    fallback = _synthesize_from_retrieved(question, state, providers)
    return _finish_result(state, fallback or state.stop("max_steps"))
