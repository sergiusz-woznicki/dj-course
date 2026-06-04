import argparse
import json
import logging
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from corpora import CORPORA_FILES  # type: ignore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    p = argparse.ArgumentParser(description="Encode corpus sentences with SBERT")
    p.add_argument("--corpus", default="WOLNELEKTURY", choices=list(CORPORA_FILES.keys()))
    p.add_argument("--model", default="sdadas/st-polish-paraphrase-from-mpnet",
                   help="HuggingFace model name or local path")
    p.add_argument("--output-dir", default="models/sbert-default")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def load_sentences(files: list) -> list[str]:
    sentences = []
    print(f"Wczytywanie z {len(files)} plików...")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                sentences.extend(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            print(f"OSTRZEŻENIE: brak pliku '{file}', pomijam.")
    if not sentences:
        raise RuntimeError("Korpus pusty.")
    return sentences


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = CORPORA_FILES[args.corpus]
    sentences = load_sentences(files)
    print(f"Wczytano {len(sentences)} zdań.")

    print(f"Ładowanie modelu: {args.model}...")
    model = SentenceTransformer(args.model)

    print(f"Enkodowanie {len(sentences)} zdań (batch={args.batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        sentences,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"Enkodowanie zakończone: {elapsed:.1f}s")

    emb_path = os.path.join(args.output_dir, "sentence_embeddings.npy")
    sentences_path = os.path.join(args.output_dir, "sentences.json")
    meta_path = os.path.join(args.output_dir, "meta.json")

    np.save(emb_path, embeddings)
    with open(sentences_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False)

    meta = {
        "model": args.model,
        "corpus": args.corpus,
        "sentence_count": len(sentences),
        "embedding_dim": embeddings.shape[1],
        "encode_time_s": round(elapsed, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nZapisano do: {args.output_dir}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Czas: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
