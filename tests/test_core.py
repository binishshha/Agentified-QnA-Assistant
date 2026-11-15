import pytest

import app.agent as agent
from app.config import Settings
from app.document_ingestion import chunk_text
from app.schemas import AskResponse
from app.tools import _calculate, execute_tool, tool_definitions


def test_chunking_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_text(text, "a.txt", size=800, overlap=120)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert chunks[1].text.split()[0] == "w680"
    assert chunks[-1].text.split()[-1] == "w1999"


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", "a.txt", size=10, overlap=10)


def test_calculator_and_safety():
    assert _calculate("2 + 3 * 4") == 14
    with pytest.raises(ValueError):
        _calculate("__import__('os').system('ls')")


def test_unknown_tool():
    with pytest.raises(ValueError):
        execute_tool("nope", {}, None, None)


def test_context_tools_record_ask_and_finish(tmp_path, monkeypatch):
    ledger = tmp_path / "evidence.jsonl"
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(ledger))
    recorded = execute_tool(
        "record_evidence",
        {"claim": "Pro costs $50", "source": "northstar_pro.md", "supports": True},
        None,
        None,
    )
    assert recorded["supports"] is True
    assert '"claim": "Pro costs $50"' in ledger.read_text(encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt: "Use the current plan")
    assert execute_tool("ask_user", {"question": "Which plan?"}, None, None)["answer"] == (
        "Use the current plan"
    )
    result = execute_tool(
        "finish",
        {
            "answer": "The plan document says $50.",
            "sources": ["northstar_pro.md"],
            "confidence": 0.9,
            "status": "answered",
        },
        None,
        None,
    )
    assert result["status"] == "answered"


def test_new_tools_are_exposed():
    names = {definition["function"]["name"] for definition in tool_definitions()}
    assert {"record_evidence", "ask_user", "finish"} <= names


def test_agent_records_trace_and_finishes_from_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_LEDGER_PATH", str(tmp_path / "evidence.jsonl"))
    responses = [
        {
            "usage": {"total_tokens": 12},
            "choices": [{"message": {"tool_calls": [{
                "id": "1",
                "function": {
                    "name": "record_evidence",
                    "arguments": '{"claim":"Pro costs $50","source":"northstar_pro.md","supports":true}',
                },
            }]}}],
        },
        {
            "usage": {"total_tokens": 8},
            "choices": [{"message": {"tool_calls": [{
                "id": "2",
                "function": {
                    "name": "finish",
                    "arguments": '{"answer":"Pro costs $50.","sources":["northstar_pro.md"],"confidence":0.9,"status":"answered"}',
                },
            }]}}],
        },
    ]

    class FakeLLM:
        store = None
        embed = None

        @staticmethod
        def _chat_kwargs():
            return {"model": "fake"}

    monkeypatch.setattr(agent, "completion", lambda **kwargs: responses.pop(0))
    result = agent.run_agent("What does Pro cost?", agent.AgentState(), FakeLLM(), max_steps=3)
    assert result["status"] == "answered"
    assert result["tokens"] == 20
    assert result["stop_reason"] == "finish"
    assert [item["tool"] for item in result["trace"]] == ["record_evidence", "finish"]


def test_agent_retries_completion_with_configured_fallback(monkeypatch):
    class Provider:
        def __init__(self, model):
            self.settings = type("Settings", (), {"llm_provider": model})()
            self.store = None

        def _chat_kwargs(self):
            return {"model": self.settings.llm_provider}

    class Router:
        primary = Provider("openai")
        fallback = Provider("ollama")
        store = None
        embed = None

    seen_models = []

    def completion(**kwargs):
        seen_models.append(kwargs["model"])
        if len(seen_models) == 1:
            raise RuntimeError("invalid API key")
        return {
            "usage": {"total_tokens": 3},
            "choices": [{"message": {"tool_calls": [{
                "id": "fallback-finish",
                "function": {
                    "name": "finish",
                    "arguments": '{"answer":"Fallback worked.","sources":[],"confidence":0.7,"status":"answered"}',
                },
            }]}}],
        }

    monkeypatch.setattr(agent, "completion", completion)
    result = agent.run_agent("Can you answer?", agent.AgentState(), Router(), max_steps=2)
    assert seen_models == ["openai", "ollama"]
    assert result["status"] == "answered"
    assert any(item["tool"] == "completion_fallback" for item in result["trace"])


def test_schema_validation():
    ok = AskResponse.model_validate_json('{"answer":"x","confidence":0.5}')
    assert ok.sources == []
    with pytest.raises(Exception):
        AskResponse.model_validate_json('{"answer":"x","confidence":2}')


def test_blank_env_values_become_none():
    s = Settings(openai_api_key=" ", fallback_provider=" ", _env_file=None)
    assert s.openai_api_key is None and s.fallback_provider is None
