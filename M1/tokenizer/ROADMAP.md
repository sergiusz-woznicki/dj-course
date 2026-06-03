# Roadmap: Benchmark tokenizerów krzyżowy

## Cel
Porównanie tokenizerów krzyżowo — 3 teksty × wszystkie tokenizery z `tokenizers/*.json`.
Pytanie: który tokenizer jest najefektywniejszy (najmniej tokenów) dla każdego tekstu?

## Teksty źródłowe

| Tekst | Źródło |
|---|---|
| Pan Tadeusz Księga 1 | `../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt` |
| The Pickwick Papers | `../korpus-mini/the-pickwick-papers-gutenberg.txt` |
| Fryderyk Chopin | `../korpus-mini/fryderyk-chopin-wikipedia.txt` |

## Tokenizery
Wszystkie pliki `tokenizers/*.json` — ładowane dynamicznie przez `glob`.

Dostępne:
- `bielik-v1-tokenizer.json`
- `bielik-v2-tokenizer.json`
- `bielik-v3-tokenizer.json`
- `QwenQwen2.5-Coder-7B-Instruct-tokenizer.json`
- `latarnik_tokenizer.json`
- `tokenizer-pan-tadeusz.json`
- `tokenizer-wolnelektury.json`
- `tokenizer-nkjp.json`
- `tokenizer-all-corpora.json`

## Metryki (tekst × tokenizer)
- `token_count` — łączna liczba tokenów
- `avg_token_len` — średnia długość tokenu = `len(text) / token_count` (wyższa = efektywniejszy)
- `unique_tokens` — liczba unikalnych tokenów

## Eksperyment: vocab_size

Dodać `--vocab-size` do `tokenizer-build.py`, wytrenować warianty ALL:

| Wariant | Komenda |
|---|---|
| 8k | `python tokenizer-build.py --corpus ALL --vocab-size 8000 --output tokenizers/tokenizer-all-8k.json` |
| 16k | `python tokenizer-build.py --corpus ALL --vocab-size 16000 --output tokenizers/tokenizer-all-16k.json` |
| 32k | już istnieje (`tokenizer-all-corpora.json`) |
| 64k | `python tokenizer-build.py --corpus ALL --vocab-size 64000 --output tokenizers/tokenizer-all-64k.json` |

## Nowy skrypt: `tokenize-benchmark.py`

Logika:
1. `glob tokenizers/*.json` → lista tokenizerów
2. Załaduj 3 teksty z plików lokalnych
3. Dla każdej pary tekst × tokenizer: encode → metryki
4. Tabela ASCII z wynikami
5. Per tekst: ranking po `token_count ASC`

Wynik końcowy: który tokenizer najefektywniejszy per tekst + porównanie własne vs bieliki.

## Kolejność wykonania
1. Dodaj `--vocab-size` do `tokenizer-build.py`
2. Wytrenuj warianty 8k/16k/64k
3. Napisz `tokenize-benchmark.py`
4. Uruchom benchmark

## Spodziewane wyniki
1. Własny tokenizer (ALL/WOLNELEKTURY) — najefektywniejszy dla PL tekstów
2. Bielik v3 — drugi
3. Bielik v1/v2 — trzeci (ten sam tokenizer)
