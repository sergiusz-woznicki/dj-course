# Plan: Zadanie 4.2 - Doc2Vec split train/infer

## Context
`run-doc2vec.py` łączy trening i wnioskowanie w jednym skrypcie. Zadanie wymaga rozdzielenia + optymalizacji parametrów. Wzorzec do naśladowania: `train-cbow.py` / `infer-cbow.py`.

## Pliki do stworzenia

### `train-doc2vec.py`
Wzorowany na `train-cbow.py`:
- `argparse`: `--tokenizer`, `--corpus`, `--vector-size`, `--window`, `--min-count`, `--workers`, `--epochs`, `--dm`, `--output-dir`
- Wczytuje korpus, tokenizuje, tworzy `TaggedDocument`
- Trenuje `Doc2Vec`, zapisuje:
  - `{output-dir}/doc2vec.model`
  - `{output-dir}/sentence_map.json` (lista raw_sentences)
  - `{output-dir}/meta.json` (parametry treningu + czas)

### `infer-doc2vec.py`
Wzorowany na `infer-cbow.py`:
- `argparse`: `--model-dir` (required), `--sentence` (required), `--topn` (default=5)
- Ładuje model + sentence_map z `--model-dir`
- Tokenizuje zdanie wejściowe tym samym tokenizerem (info z meta.json)
- `model.infer_vector(tokens)` → `dv.most_similar([vec], topn=topn)`
- Wypisuje wyniki w formacie: `Sim: X.XXXX | Zdanie: ...`

## Pliki do zmiany
- `run-doc2vec.py` — bez zmian (zostaje jako legacy/reference)

## Krytyczne pliki referencyjne
- `train-cbow.py` — wzorzec argparse + save artifacts
- `infer-cbow.py` — wzorzec load + query
- `corpora.py` — `CORPORA_FILES` dict

## Status
- [x] `train-doc2vec.py` — utworzony
- [x] `infer-doc2vec.py` — utworzony
- [x] Fix: filtr krótkich zdań w korpusie (`--min-sentence-len`, problem "FAUST")
- [x] Eksperyment DBOW (dm=0) — 3× szybszy, lepsza jakość semantyczna

## Problem: wyniki "FAUST"
Korpus WOLNELEKTURY zawiera dramaty (np. Faust) gdzie nazwy mówców są osobnymi liniami ("FAUST", "MEFISTOFELES"). Model trenuje na nich i zwraca je jako "podobne zdania".

Fix w `train-doc2vec.py`: dodać `--min-sentence-len` (int, default=5) — filtruje linie krótsze niż N słów przy `load_sentences()`.

## Weryfikacja
```bash
# trening
python train-doc2vec.py --corpus WOLNELEKTURY --output-dir models/d2v-test --epochs 20

# wnioskowanie
python infer-doc2vec.py --model-dir models/d2v-test --sentence "Jestem głodny i bardzo chętnie zjadłbym coś." --topn 5

# test ze zdaniem z korpusu (powinno mieć sim ~1.0)
python infer-doc2vec.py --model-dir models/d2v-test --sentence "Rzędzian siedział ciągle pod oknem." --topn 5
```
