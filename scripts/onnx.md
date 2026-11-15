# Model Optimization and ONNX Architecture Analysis

## Decision

ONNX conversion is not appropriate for the chat models in this project. It is
appropriate as an optional optimization for the local RAG embedding model.

| Component | Runtime in this project | ONNX applicability | Recommended optimization |
| --- | --- | --- | --- |
| Primary chat model | OpenAI hosted API | Not applicable | Response caching, retries, rate limiting, and asynchronous request handling |
| Fallback chat model | Ollama with a GGUF model | Usually not applicable | GGUF quantization through Ollama/llama.cpp |
| Optional GPU chat model | vLLM | Usually not applicable | vLLM serving and GPU-oriented batching |
| RAG embeddings | `sentence-transformers/all-MiniLM-L6-v2` on CPU | Applicable | ONNX Runtime and optional INT8 quantization |

## Chat models

### OpenAI primary model

The primary assistant model is accessed through OpenAI's hosted API. The model
weights and computation graph remain on OpenAI's infrastructure, so this
project cannot export them to ONNX.

The relevant optimizations happen at the application boundary:

- Response caching in `app/cache.py`.
- Retry and exponential backoff in `app/resilience.py`.
- Provider fallback and degraded retrieval in `app/llm_router.py`.
- Rate limiting in `app/rate_limit.py`.
- Async FastAPI routes with blocking model work moved to a thread pool in
  `app/main.py`.

### Ollama fallback

The CPU fallback uses Ollama, which normally serves quantized GGUF models
through llama.cpp. GGUF is already designed for compact local inference and
CPU-friendly quantization. Converting that served model to ONNX would not
normally improve this deployment and could lose optimizations provided by the
Ollama/llama.cpp runtime.

### Optional vLLM provider

vLLM is an optional provider for a GPU or remote model server. It is optimized
for serving generative LLMs with batching and efficient KV-cache management.
ONNX is therefore not the preferred optimization path for this component
either.

## RAG embedding model

The application uses `sentence-transformers/all-MiniLM-L6-v2` to embed both
uploaded document chunks and user questions. This is an encoder-only model,
not an autoregressive chat model. It processes each input in a forward pass,
which makes it a reasonable ONNX Runtime candidate for CPU inference.

The current path is:

```text
document or question
    -> SentenceTransformer on CPU
    -> embedding vector
    -> ChromaDB index/search
```

The export utility is [`export_onnx.py`](./export_onnx.py). It creates ONNX
files but does not change the runtime used by `app/llm_provider.py`.

## Potential benefits

ONNX Runtime may provide the following benefits for the embedding workload:

- Lower memory use after graph optimization or INT8 quantization.
- Faster CPU embedding for large document ingestion workloads.
- A smaller, more predictable inference runtime for deployment.

These benefits are workload- and hardware-dependent. The project does not
currently include benchmark results, so claimed speedups must be measured on
the target laptop or deployment machine.

## Export commands

Install the optional tools:

```powershell
pip install "optimum[onnxruntime]" transformers
```

Export the embedding model:

```powershell
python scripts/export_onnx.py
```

Export an INT8 version:

```powershell
python scripts/export_onnx.py --quantize
```

The default output directory is `models/minilm-onnx/`.

## Adoption requirements

Exporting the model alone is not enough. To use ONNX in the application:

1. Add an ONNX Runtime embedding implementation in `app/llm_provider.py`.
2. Ensure tokenization, pooling, normalization, and output dimensions match
   the existing SentenceTransformer behavior.
3. Compare retrieval quality against the current implementation.
4. Benchmark ingestion and query embedding latency on the target CPU.
5. Delete `data/chroma_db/` and re-index all documents after switching the
   embedding implementation.

The ChromaDB index must be rebuilt because vector indexes depend on the exact
embedding model and preprocessing pipeline that created their vectors.

## Benchmark recommendation

Benchmark the current PyTorch/SentenceTransformer path against ONNX Runtime
using the same document set, batch size, and CPU. Record:

- Cold-start time.
- Embedding latency for one query.
- Batch ingestion throughput.
- Peak memory usage.
- Retrieval quality and cosine similarity consistency.

Adopt ONNX only if it improves the target workload without reducing retrieval
quality. For the current application, ONNX is an optional embedding
optimization, not a replacement for OpenAI, Ollama, or vLLM chat serving.
