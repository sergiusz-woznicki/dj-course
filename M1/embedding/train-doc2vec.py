import argparse
import json
import logging
import os
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tokenizers import Tokenizer
from corpora import CORPORA_FILES  # type: ignore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TOKENIZERS_DIR = "../tokenizer/tokenizers"


def parse_args():
    p = argparse.ArgumentParser(description="Train Doc2Vec model")
    p.add_argument("--tokenizer", default="bielik-v3-tokenizer", help="Tokenizer name (without .json)")
    p.add_argument("--corpus", default="WOLNELEKTURY", choices=list(CORPORA_FILES.keys()))
    p.add_argument("--vector-size", type=int, default=100)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--min-count", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--dm", type=int, default=1, choices=[0, 1], help="1=PV-DM, 0=PV-DBOW")
    p.add_argument("--output-dir", default="models/doc2vec-default")
    p.add_argument("--min-sentence-len", type=int, default=5, help="Minimalna liczba słów w zdaniu")
    return p.parse_args()


def load_sentences(files: list, min_words: int = 5) -> list[str]:
    raw = []
    print(f"Wczytywanie {len(files)} plików...")
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                raw.extend(line.strip() for line in f if line.strip() and len(line.split()) >= min_words)
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
    raw_sentences = load_sentences(files, min_words=args.min_sentence_len)

    print(f"Tokenizacja {len(raw_sentences)} zdań...")
    encodings = tokenizer.encode_batch(raw_sentences)
    tagged_data = [
        TaggedDocument(words=enc.tokens, tags=[str(i)])
        for i, enc in enumerate(encodings)
    ]
    print(f"Gotowe: {len(tagged_data)} TaggedDocument.")

    print("\n--- Trening Doc2Vec ---")
    t0 = time.time()
    model = Doc2Vec(
        tagged_data,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        epochs=args.epochs,
        dm=args.dm,
    )
    elapsed = time.time() - t0
    print(f"Trening zakończony: {elapsed:.1f}s")

    model_path = os.path.join(args.output_dir, "doc2vec.model")
    map_path = os.path.join(args.output_dir, "sentence_map.json")
    meta_path = os.path.join(args.output_dir, "meta.json")

    model.save(model_path)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(raw_sentences, f, ensure_ascii=False, indent=2)

    meta = {
        "tokenizer": args.tokenizer,
        "corpus": args.corpus,
        "vector_size": args.vector_size,
        "window": args.window,
        "min_count": args.min_count,
        "epochs": args.epochs,
        "dm": args.dm,
        "train_time_s": round(elapsed, 1),
        "num_sentences": len(raw_sentences),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nZapisano do: {args.output_dir}")
    print(f"  Zdań: {len(raw_sentences)}")
    print(f"  Czas: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
