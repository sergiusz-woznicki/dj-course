import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    p = argparse.ArgumentParser(description="Query SBERT sentence similarity")
    p.add_argument("--model-dir", required=True, help="Directory with encoded corpus")
    p.add_argument("--sentence", required=True, help="Query sentence")
    p.add_argument("--topn", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    meta_path = os.path.join(args.model_dir, "meta.json")
    emb_path = os.path.join(args.model_dir, "sentence_embeddings.npy")
    sentences_path = os.path.join(args.model_dir, "sentences.json")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    embeddings = np.load(emb_path)

    with open(sentences_path, encoding="utf-8") as f:
        sentences = json.load(f)

    model_name = meta["model"]
    print(f"Model: {model_name}")
    print(f"Korpus: {meta['corpus']} ({meta['sentence_count']} zdań, dim={meta['embedding_dim']})")

    model = SentenceTransformer(model_name)

    query_emb = model.encode([args.sentence], convert_to_numpy=True)
    sims = cosine_similarity(query_emb, embeddings)[0]

    top_indices = np.argsort(sims)[::-1][:args.topn]

    print(f"\nZapytanie: \"{args.sentence}\"")
    print(f"{args.topn} najbardziej podobnych zdań:")
    for i in top_indices:
        print(f"  - Sim: {sims[i]:.4f} | Zdanie: {sentences[i]}")


if __name__ == "__main__":
    main()
