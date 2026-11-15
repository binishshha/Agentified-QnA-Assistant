# Agentic research architecture

```mermaid
flowchart TD
    User --> Research[POST /research]
    Research --> Loop[Single-agent loop\nmax 8 steps / 20,000 tokens]
    Loop --> LLM[LLM decides next action]
    LLM --> Tools[Search / metadata / calculator]
    Tools --> Store[(Chroma document store)]
    Tools --> Notes[(JSONL evidence ledger)]
    Notes --> Prune[Prune old tool results\nkeep last 2 + inject notes]
    Prune --> LLM
    LLM --> Evidence[record_evidence]
    Evidence --> Notes
    LLM --> Finish[finish / ask_user]
    Finish --> Response[Answer + status + trace + tokens]
```

The loop is single-agent because searches, conflict resolution, and evidence
recording share one tightly coupled ledger. Search calls are fast and do not
create a useful parallelization opportunity. The main remaining risk is the
self-verification paradox: the same model records evidence and judges whether
it is sufficient. The deterministic citation check reduces that risk by
downgrading any answer whose cited source is absent from the ledger.
