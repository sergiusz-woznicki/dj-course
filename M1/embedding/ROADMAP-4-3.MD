# Plan: Zadanie 4.3 - SBERT split encode/infer + lepszy model PL

## Cel
1. Rozdziel `run-sbert.py` na dwa skrypty: `encode-sbert.py` (kodowanie bazy) i `infer-sbert.py` (odpytywanie)
2. Znajdź lepszy model dla polskiego (zamiennik `intfloat/multilingual-e5-small`)
3. Przetestuj z zdaniami wymyślonymi i z korpusu

## Model PL — kandydaci
- `sdadas/st-polish-paraphrase-from-mpnet` — finetuned na PL, paraphrase
- `sdadas/st-polish-paraphrase-from-distilroberta` — szybszy wariant
- `allegro/herbert-base-cased` — encoder PL, ale nie sentence-transformer
- `intfloat/multilingual-e5-base` — większy wariant obecnego (baseline)

**Wybór**: `sdadas/st-polish-paraphrase-from-mpnet` — trenowany na polskich parach zdań, dedykowany do wyszukiwania podobieństwa semantycznego.

## Pliki do stworzenia

### `encode-sbert.py`
- `argparse`: `--corpus`, `--model`, `--output-dir`
- Wczytuje zdania z korpusu (identycznie jak `run-sbert.py`)
- Enkoduje → `sentence_embeddings.npy` + `sentences.json` (lista surowych zdań)
- Zapisuje `meta.json` (model, corpus, count, czas)
- Output dir default: `models/sbert-default`

### `infer-sbert.py`
- `argparse`: `--model-dir` (required), `--sentence` (required), `--topn` (default=5)
- Wczytuje `sentence_embeddings.npy` + `sentences.json` + `meta.json` z `--model-dir`
- Ładuje model SBERT (nazwa z meta.json)
- Enkoduje zapytanie → cosine similarity → top-N
- Format: `Sim: X.XXXX | Zdanie: ...`

## Wzorzec
- `train-cbow.py` / `infer-cbow.py` — struktura argparse + meta.json

## Weryfikacja
```bash
# kodowanie bazy
python encode-sbert.py --corpus WOLNELEKTURY --model sdadas/st-polish-paraphrase-from-mpnet --output-dir models/sbert-pl

# zapytanie wymyślone
python infer-sbert.py --model-dir models/sbert-pl --sentence "Jestem głodny i chętnie coś zjem." --topn 5

# zapytanie z korpusu (sim ~1.0)
python infer-sbert.py --model-dir models/sbert-pl --sentence "Rzędzian siedział ciągle pod oknem." --topn 5

# porównanie modeli
python encode-sbert.py --corpus WOLNELEKTURY --model intfloat/multilingual-e5-small --output-dir models/sbert-e5
python infer-sbert.py --model-dir models/sbert-e5 --sentence "Jestem głodny i chętnie coś zjem." --topn 5
```

## Status
- [x] `encode-sbert.py` — created
- [x] `infer-sbert.py` — created
- [x] Enkodowanie PAN_TADEUSZ z modelem PL (198.8s, 10061 zdań)
- [x] Enkodowanie PAN_TADEUSZ z baseline e5-small (34.6s)
- [x] Testy zapytań — wymyślone + z korpusu
- [x] Wyniki do RESULTS-4-3.md
