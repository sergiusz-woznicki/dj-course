# Plan: Zadanie 4.1 — CBOW refactor + experiments

## Context
`run-cbow.py` robi wszystko naraz: ładuje dane, trenuje, zapisuje, inferuje. Zadanie wymaga rozdziału na train/infer + eksperymenty z tokenizerami i parametrami.

## Zmiany

### 1. Rozdziel `run-cbow.py` na dwa skrypty

**`train-cbow.py`** — tylko trening, zapis modelu:
- Args CLI: `--tokenizer`, `--corpus`, `--vector-size`, `--window`, `--min-count`, `--epochs`, `--sample-rate`, `--output-dir`
- Ładuje tokenizer z `../tokenizer/tokenizers/<name>.json`
- Ładuje korpus przez `corpora.py` (CORPORA_FILES[corpus])
- Trenuje Word2Vec CBOW
- Zapisuje: model `.model`, embeddings `.npy`, token map `.json` — do `--output-dir` (np. `models/`)
- Loguje czas treningu

**`infer-cbow.py`** — tylko wnioskowanie:
- Args CLI: `--model-dir`, `--words` (lista słów, lub kombinacja)
- Ładuje model z `--model-dir`
- Wywołuje `get_word_vector_and_similar()` dla każdego słowa/kombinacji
- Wypisuje top-N podobnych

### 2. Eksperymenty z tokenizerami

Trzy tokenizery wg RESULTS.md (PL corpus):
- `tokenizer-all-64k` — **najlepszy dla PL** (śr.dł. 4.64, najmniej tokenów) → custom z zad.3
- `bielik-v3-tokenizer` — najlepszy z bielików dla PL
- `bielik-v1-tokenizer` — per homework "porównaj bielik-v1 (mistralowy)" — spodziewamy się najgorszy (śr.dł. 2.14)

Słowa testowe: `wojsko`, `szlachta`, `choroba`, `król` + kombinacja `['dziecko', 'kobieta']`

### 3. Domyślne parametry startowe (z obecnego skryptu)
```
VECTOR_LENGTH = 20
WINDOW_SIZE = 6
MIN_COUNT = 2
WORKERS = 4
EPOCHS = 20
SAMPLE_RATE = 1e-2
```

## Pliki do modyfikacji
- `M1/embedding/run-cbow.py` → podziel na dwa nowe pliki, stary można usunąć lub zachować jako legacy
- Nowe: `M1/embedding/train-cbow.py`, `M1/embedding/infer-cbow.py`
- `M1/embedding/corpora.py` — bez zmian

## Weryfikacja
```bash
# Trening — custom 64k (najlepszy PL)
python train-cbow.py --tokenizer tokenizer-all-64k --corpus WOLNELEKTURY --output-dir models/all-64k/
python infer-cbow.py --model-dir models/all-64k/ --words wojsko szlachta choroba król
python infer-cbow.py --model-dir models/all-64k/ --words dziecko kobieta --combine

# Trening — bielik-v3
python train-cbow.py --tokenizer bielik-v3-tokenizer --corpus WOLNELEKTURY --output-dir models/bielik-v3/
python infer-cbow.py --model-dir models/bielik-v3/ --words wojsko król

# Trening — bielik-v1 (spodziewany najgorszy)
python train-cbow.py --tokenizer bielik-v1-tokenizer --corpus WOLNELEKTURY --output-dir models/bielik-v1/
python infer-cbow.py --model-dir models/bielik-v1/ --words wojsko król
```

Oczekiwany wynik: `król–książę` similarity ≥ 0.7, `dziecko+kobieta→dziewczyna` ≥ 0.6.
