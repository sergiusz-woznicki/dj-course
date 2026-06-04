import argparse
import json
import os
from gensim.models.doc2vec import Doc2Vec
from tokenizers import Tokenizer

TOKENIZERS_DIR = "../tokenizer/tokenizers"


def parse_args():
    p = argparse.ArgumentParser(description="Doc2Vec inference: find similar sentences")
    p.add_argument("--model-dir", required=True, help="Dir from train-doc2vec.py")
    p.add_argument("--sentence", required=True, help="Query sentence")
    p.add_argument("--topn", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    meta_path = os.path.join(args.model_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    tokenizer_path = f"{TOKENIZERS_DIR}/{meta['tokenizer']}.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    model_path = os.path.join(args.model_dir, "doc2vec.model")
    model = Doc2Vec.load(model_path)

    map_path = os.path.join(args.model_dir, "sentence_map.json")
    with open(map_path, encoding="utf-8") as f:
        sentence_map: list[str] = json.load(f)

    tokens = tokenizer.encode(args.sentence).tokens
    vector = model.infer_vector(tokens, epochs=model.epochs)

    similar = model.dv.most_similar([vector], topn=args.topn)

    print(f'\nZdanie do wnioskowania: "{args.sentence}"')
    print(f"{args.topn} najbardziej podobnych zdań z korpusu:")
    for doc_id_str, sim in similar:
        idx = int(doc_id_str)
        sentence = sentence_map[idx] if idx < len(sentence_map) else f"[BRAK ID: {doc_id_str}]"
        print(f"  - Sim: {sim:.4f} | Zdanie: {sentence}")


if __name__ == "__main__":
    main()
