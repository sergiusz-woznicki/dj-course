"""
Run CBOW experiments across tokenizers, corpora, and params.
Outputs markdown table rows to stdout — redirect to file or capture.
"""
import subprocess
import json
import os
import sys
import time

TOKENIZERS = ["tokenizer-all-64k", "bielik-v3-tokenizer", "bielik-v1-tokenizer"]
CORPORA = ["WOLNELEKTURY", "ALL"]
BASE_PARAMS = dict(vector_size=20, window=6, min_count=2, epochs=20, sample_rate=1e-2)

# Param sweep on best config (determined after phase 1)
PARAM_SWEEP = [
    dict(vector_size=20,  epochs=10),
    dict(vector_size=20,  epochs=20),  # baseline
    dict(vector_size=20,  epochs=50),
    dict(vector_size=50,  epochs=20),
    dict(vector_size=100, epochs=20),
    dict(vector_size=100, epochs=50),
]
PARAM_SWEEP_TOKENIZER = "tokenizer-all-64k"
PARAM_SWEEP_CORPUS = "WOLNELEKTURY"

TEST_WORDS = ["wojsko", "szlachta", "choroba", "król"]
COMBINE_WORDS = ["dziecko", "kobieta"]

def train(tag: str, tokenizer: str, corpus: str, **params) -> str:
    out_dir = f"models/exp/{tag}"
    cmd = [
        "python3", "train-cbow.py",
        "--tokenizer", tokenizer,
        "--corpus", corpus,
        "--output-dir", out_dir,
        "--vector-size", str(params.get("vector_size", 20)),
        "--window", str(params.get("window", 6)),
        "--min-count", str(params.get("min_count", 2)),
        "--epochs", str(params.get("epochs", 20)),
        "--sample-rate", str(params.get("sample_rate", 1e-2)),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [BŁĄD train] {tag}:\n{result.stderr[-500:]}", file=sys.stderr)
    return out_dir

def infer_similarity(model_dir: str, words: list[str], combine=False) -> dict[str, float] | None:
    cmd = ["python3", "infer-cbow.py", "--model-dir", model_dir, "--words"] + words
    if combine:
        cmd.append("--combine")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    similarities = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("- ") and ": " in line:
            parts = line[2:].rsplit(": ", 1)
            if len(parts) == 2:
                try:
                    similarities[parts[0].strip()] = float(parts[1])
                except ValueError:
                    pass
    return similarities

def meta_time(model_dir: str) -> float:
    try:
        with open(os.path.join(model_dir, "meta.json")) as f:
            return json.load(f).get("train_time_s", 0)
    except Exception:
        return 0

def top3_str(sims: dict | None) -> str:
    if not sims:
        return "brak"
    top = list(sims.items())[:3]
    return ", ".join(f"{t}:{s:.3f}" for t, s in top)

def get_sim(sims: dict | None, word: str) -> str:
    if not sims:
        return "-"
    return f"{sims.get(word, 0):.4f}" if word in sims else "-"

# ── Phase 1: tokenizer × corpus ────────────────────────────────────────────
print("\n## Faza 1: Tokenizer × Korpus (parametry domyślne)\n")
print("| Tokenizer | Korpus | Czas(s) | król–książę | dziecko+kobieta top1 |")
print("|---|---|---|---|---|")

phase1_results = []
for tok in TOKENIZERS:
    for corp in CORPORA:
        tag = f"{tok}__{corp}__base"
        print(f"  Trening: {tag}...", file=sys.stderr)
        out_dir = train(tag, tok, corp, **BASE_PARAMS)
        t = meta_time(out_dir)

        krol_sims = infer_similarity(out_dir, ["król"])
        ksiaze = get_sim(krol_sims, "książę") if krol_sims else "-"

        comb_sims = infer_similarity(out_dir, COMBINE_WORDS, combine=True)
        comb_top1 = top3_str({k: v for k, v in (comb_sims or {}).items()
                               if k not in COMBINE_WORDS}[:1].items() if False else
                              dict(list({k: v for k, v in (comb_sims or {}).items()
                                         if k not in COMBINE_WORDS}.items())[:1]))

        row = dict(tok=tok, corp=corp, time=t, ksiaze=ksiaze, comb_top1=comb_top1)
        phase1_results.append(row)
        print(f"| {tok} | {corp} | {t} | {ksiaze} | {comb_top1} |")

# ── Phase 2: param sweep on best config ────────────────────────────────────
print(f"\n## Faza 2: Param sweep ({PARAM_SWEEP_TOKENIZER} × {PARAM_SWEEP_CORPUS})\n")
print("| vector_size | epochs | Czas(s) | król–książę | dziecko+kobieta top1 |")
print("|---|---|---|---|---|")

for p in PARAM_SWEEP:
    tag = f"sweep__vs{p['vector_size']}__ep{p['epochs']}"
    print(f"  Trening: {tag}...", file=sys.stderr)
    params = {**BASE_PARAMS, **p}
    out_dir = train(tag, PARAM_SWEEP_TOKENIZER, PARAM_SWEEP_CORPUS, **params)
    t = meta_time(out_dir)

    krol_sims = infer_similarity(out_dir, ["król"])
    ksiaze = get_sim(krol_sims, "książę") if krol_sims else "-"

    comb_sims = infer_similarity(out_dir, COMBINE_WORDS, combine=True)
    comb_top1 = top3_str(dict(list({k: v for k, v in (comb_sims or {}).items()
                                     if k not in COMBINE_WORDS}.items())[:1]))

    print(f"| {p['vector_size']} | {p['epochs']} | {t} | {ksiaze} | {comb_top1} |")
