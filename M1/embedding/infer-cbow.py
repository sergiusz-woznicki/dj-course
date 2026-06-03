import argparse
import json
import os
import numpy as np
from gensim.models import Word2Vec
from tokenizers import Tokenizer

TOKENIZERS_DIR = "../tokenizer/tokenizers"

def parse_args():
    p = argparse.ArgumentParser(description="CBOW inference: find similar words")
    p.add_argument("--model-dir", required=True, help="Dir from train-cbow.py")
    p.add_argument("--words", nargs="+", required=True, help="Words to query")
    p.add_argument("--combine", action="store_true", help="Combine all words into one query vector")
    p.add_argument("--topn", type=int, default=10)
    return p.parse_args()

def get_word_vector(word: str, tokenizer: Tokenizer, model: Word2Vec) -> np.ndarray | None:
    encoding = tokenizer.encode(" " + word + " ")
    tokens = [t.strip() for t in encoding.tokens if t.strip()]
    tokens = [t for t in tokens if t not in ('[CLS]', '<s>', '[SEP]', '</s>', 'Ġ')]

    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        missing = [t for t in tokens if t not in model.wv]
        print(f"  BŁĄD: brak wektorów dla '{word}' (tokeny: {tokens}, brak w vocab: {missing})")
        return None
    return np.mean(vectors, axis=0)

def main():
    args = parse_args()

    meta_path = os.path.join(args.model_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    tokenizer_path = f"{TOKENIZERS_DIR}/{meta['tokenizer']}.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = Word2Vec.load(os.path.join(args.model_dir, "model.model"))

    print(f"Model: {args.model_dir} | tokenizer={meta['tokenizer']} corpus={meta['corpus']} epochs={meta['epochs']} vecsize={meta['vector_size']}")

    if args.combine:
        vecs = []
        for word in args.words:
            v = get_word_vector(word, tokenizer, model)
            if v is not None:
                vecs.append(v)
        if not vecs:
            print("Brak wektorów do kombinacji.")
            return
        combined = np.mean(vecs, axis=0)
        similar = model.wv.most_similar(positive=[combined], topn=args.topn)
        print(f"\n{args.topn} najbardziej podobnych do kombinacji {args.words}:")
        for token, sim in similar:
            print(f"  - {token}: {sim:.4f}")
    else:
        for word in args.words:
            vec = get_word_vector(word, tokenizer, model)
            if vec is None:
                continue
            similar = model.wv.most_similar(positive=[vec], topn=args.topn)
            tokens_used = tokenizer.encode(" " + word + " ").tokens
            print(f"\n{args.topn} najbardziej podobnych do '{word}' (tokeny: {tokens_used}):")
            print(f"  > Wektor (początek): {vec[:5]}...")
            for token, sim in similar:
                print(f"  - {token}: {sim:.4f}")

if __name__ == "__main__":
    main()
