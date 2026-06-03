from pathlib import Path
from tokenizers import Tokenizer

TEXTS = {
    "Pan Tadeusz Ks.1": "../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt",
    "Pickwick Papers": "../korpus-mini/the-pickwick-papers-gutenberg.txt",
    "Fryderyk Chopin": "../korpus-mini/fryderyk-chopin-wikipedia.txt",
}

def load_tokenizers() -> dict[str, Tokenizer]:
    tokenizers = {}
    for path in sorted(Path("tokenizers").glob("*.json")):
        try:
            tokenizers[path.stem] = Tokenizer.from_file(str(path))
        except Exception as e:
            print(f"[SKIP] {path.stem}: {e}")
    return tokenizers

def load_texts() -> dict[str, str]:
    texts = {}
    for name, path in TEXTS.items():
        with open(path, encoding="utf-8") as f:
            texts[name] = f.read()
    return texts

def benchmark(texts: dict[str, str], tokenizers: dict[str, Tokenizer]) -> dict:
    results = {}
    for tname, text in texts.items():
        results[tname] = {}
        for tok_name, tok in tokenizers.items():
            enc = tok.encode(text)
            count = len(enc.ids)
            avg_len = len(text) / count if count else 0
            unique = len(set(enc.ids))
            results[tname][tok_name] = {
                "token_count": count,
                "avg_token_len": round(avg_len, 2),
                "unique_tokens": unique,
            }
    return results

def print_table(results: dict) -> None:
    for text_name, tok_results in results.items():
        print(f"\n{'='*70}")
        print(f"Tekst: {text_name}")
        print(f"{'='*70}")
        ranked = sorted(tok_results.items(), key=lambda x: x[1]["token_count"])
        print(f"{'#':<3} {'Tokenizer':<42} {'Tokeny':>8} {'Śr.dł.':>7} {'Unique':>8}")
        print(f"{'-'*3} {'-'*42} {'-'*8} {'-'*7} {'-'*8}")
        for rank, (tok_name, m) in enumerate(ranked, 1):
            print(f"{rank:<3} {tok_name:<42} {m['token_count']:>8} {m['avg_token_len']:>7} {m['unique_tokens']:>8}")
        best = ranked[0][0]
        print(f"\n  NAJEFEKTYWNIEJSZY: {best} ({ranked[0][1]['token_count']} tokenów)")

def main() -> None:
    print("Ładowanie tokenizerów...")
    tokenizers = load_tokenizers()
    print(f"Załadowano {len(tokenizers)} tokenizerów")

    print("Ładowanie tekstów...")
    texts = load_texts()
    for name, text in texts.items():
        print(f"  {name}: {len(text):,} znaków")

    print("\nBenchmark...")
    results = benchmark(texts, tokenizers)
    print_table(results)

if __name__ == "__main__":
    main()
