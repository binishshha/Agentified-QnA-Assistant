"""Optionally export the local embedding model to ONNX.

The application currently uses SentenceTransformer directly on the CPU, so
this script is not required for normal operation. ONNX may reduce CPU memory
usage or improve embedding throughput, but it must be benchmarked first.

This script only creates ONNX files; it does not change the runtime model used
by the application. To adopt the exported model, update the embedding code in
``app/llm_provider.py``, validate the vector dimensions and search quality,
then rebuild ``data/chroma_db`` and re-index all documents.

The hosted chat model is not an ONNX candidate. ONNX is relevant here only to
the local document and question embedding model.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", default="models/minilm-onnx")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Install optimum[onnxruntime] and transformers first"
        ) from error
    model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            f"{args.out}/model.onnx",
            f"{args.out}/model.int8.onnx",
            weight_type=QuantType.QInt8,
        )
    print(f"Exported embedding model to {args.out}")


if __name__ == "__main__":
    main()
