import argparse
import json
import logging
import os
import time
import numpy as np
from gensim.models import Word2Vec
from tokenizers import Tokenizer
from corpora import CORPORA_FILES  # type: ignore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TOKENIZERS_DIR = "../tokenizer/tokenizers"

def parse_args():
    p = argparse.ArgumentParser(description="Train CBOW Word2Vec model")
    p.add_argument("--tokenizer", default="tokenizer-all-64k", help="Tokenizer name (without .json)")
    p.add_argument("--corpus", default="WOLNELEKTURY", choices=list(CORPORA_FILES.keys()))
    p.add_argument("--vector-size", type=int, default=20)
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--sample-rate", type=float, default=1e-2)
    p.add_argument("--output-dir", default="models/default")
    return p.parse_args()

def aggregate_raw_sentences(files: list[str]) -> list[str]:
    raw = []
    print(f"Wczytywanie {len(files)} plików...")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                raw.extend(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: brak pliku '{file}', pomijam.")
    if not raw:
        raise RuntimeError("Brak zdań w korpusie.")
    return raw

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_path = f"{TOKENIZERS_DIR}/{args.tokenizer}.json"
    print(f"Tokenizer: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    files = CORPORA_FILES[args.corpus]
    raw_sentences = aggregate_raw_sentences(files)

    print(f"Tokenizacja {len(raw_sentences)} zdań...")
    encodings = tokenizer.encode_batch(raw_sentences)
    tokenized = [enc.tokens for enc in encodings]
    print(f"Gotowe: {len(tokenized)} sekwencji.")

    print("\n--- Trening Word2Vec CBOW ---")
    t0 = time.time()
    model = Word2Vec(
        sentences=tokenized,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=0,
        epochs=args.epochs,
        sample=args.sample_rate,
    )
    elapsed = time.time() - t0
    print(f"Trening zakończony: {elapsed:.1f}s")

    tensor = np.array(model.wv.vectors, dtype=np.float32)
    token_to_index = {t: model.wv.get_index(t) for t in model.wv.index_to_key}

    model_path = os.path.join(args.output_dir, "model.model")
    tensor_path = os.path.join(args.output_dir, "embeddings.npy")
    map_path = os.path.join(args.output_dir, "token_map.json")

    model.save(model_path)
    np.save(tensor_path, tensor)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(token_to_index, f, ensure_ascii=False)

    # Zapisz meta dla infer-cbow
    meta = {
        "tokenizer": args.tokenizer,
        "corpus": args.corpus,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "sample_rate": args.sample_rate,
        "train_time_s": round(elapsed, 1),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nZapisano do: {args.output_dir}")
    print(f"  Tensor: {tensor.shape}")
    print(f"  Czas: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
