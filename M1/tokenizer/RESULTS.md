# Wyniki benchmarku tokenizerów

Data: 2026-06-03

## Teksty źródłowe

| Tekst | Źródło | Rozmiar |
|---|---|---|
| Pan Tadeusz Ks.1 | `../korpus-wolnelektury/pan-tadeusz-ksiega-1.txt` | 43 734 znaków |
| Pickwick Papers | `../korpus-mini/the-pickwick-papers-gutenberg.txt` | 1 746 334 znaków |
| Fryderyk Chopin | `../korpus-mini/fryderyk-chopin-wikipedia.txt` | 59 585 znaków |

## Wyniki: Pan Tadeusz Ks.1

| # | Tokenizer | Tokeny | Śr.dł. | Unique |
|---|---|---|---|---|
| 1 | tokenizer-all-64k | 9 434 | 4.64 | 3 952 |
| 2 | tokenizer-pan-tadeusz | 9 985 | 4.38 | 3 757 |
| 3 | latarnik_tokenizer | 10 045 | 4.35 | 3 853 |
| 4 | tokenizer-wolnelektury | 10 045 | 4.35 | 3 853 |
| 5 | tokenizer-all-corpora | 10 208 | 4.28 | 3 852 |
| 6 | tokenizer-nkjp | 11 066 | 3.95 | 3 574 |
| 7 | tokenizer-all-16k | 11 149 | 3.92 | 3 489 |
| 8 | tokenizer-all-8k | 12 303 | 3.55 | 2 952 |
| 9 | bielik-v3-tokenizer | 13 177 | 3.32 | 3 632 |
| 10 | QwenQwen2.5-Coder-7B-Instruct-tokenizer | 17 292 | 2.53 | 2 108 |
| 11 | bielik-v2-tokenizer | 20 480 | 2.14 | 1 461 |
| 12 | bielik-v1-tokenizer | 20 481 | 2.14 | 1 462 |

## Wyniki: Pickwick Papers

| # | Tokenizer | Tokeny | Śr.dł. | Unique |
|---|---|---|---|---|
| 1 | QwenQwen2.5-Coder-7B-Instruct-tokenizer | 445 303 | 3.92 | 16 993 |
| 2 | bielik-v2-tokenizer | 503 668 | 3.47 | 11 626 |
| 3 | bielik-v1-tokenizer | 503 669 | 3.47 | 11 627 |
| 4 | tokenizer-all-64k | 698 684 | 2.50 | 3 185 |
| 5 | tokenizer-nkjp | 725 407 | 2.41 | 2 686 |
| 6 | bielik-v3-tokenizer | 729 254 | 2.39 | 3 801 |
| 7 | tokenizer-all-corpora | 746 299 | 2.34 | 2 388 |
| 8 | tokenizer-all-16k | 816 153 | 2.14 | 1 752 |
| 9 | latarnik_tokenizer | 824 016 | 2.12 | 1 924 |
| 10 | tokenizer-wolnelektury | 824 016 | 2.12 | 1 924 |
| 11 | tokenizer-all-8k | 864 016 | 2.02 | 1 248 |
| 12 | tokenizer-pan-tadeusz | 926 941 | 1.88 | 1 043 |

## Wyniki: Fryderyk Chopin

| # | Tokenizer | Tokeny | Śr.dł. | Unique |
|---|---|---|---|---|
| 1 | tokenizer-all-64k | 13 313 | 4.48 | 4 407 |
| 2 | tokenizer-nkjp | 14 100 | 4.23 | 4 296 |
| 3 | tokenizer-all-corpora | 14 722 | 4.05 | 4 170 |
| 4 | bielik-v3-tokenizer | 16 338 | 3.65 | 4 418 |
| 5 | tokenizer-all-16k | 16 469 | 3.62 | 3 726 |
| 6 | latarnik_tokenizer | 16 917 | 3.52 | 3 740 |
| 7 | tokenizer-wolnelektury | 16 917 | 3.52 | 3 740 |
| 8 | tokenizer-all-8k | 18 311 | 3.25 | 3 085 |
| 9 | tokenizer-pan-tadeusz | 20 337 | 2.93 | 2 719 |
| 10 | QwenQwen2.5-Coder-7B-Instruct-tokenizer | 22 438 | 2.66 | 2 939 |
| 11 | bielik-v2-tokenizer | 25 610 | 2.33 | 2 170 |
| 12 | bielik-v1-tokenizer | 25 611 | 2.33 | 2 171 |

## Najefektywniejszy per tekst

| Tekst | Najefektywniejszy tokenizer | Tokeny | Śr.dł. |
|---|---|---|---|
| Pan Tadeusz Ks.1 (PL) | `tokenizer-all-64k` | 9 434 | 4.64 |
| Pickwick Papers (EN) | `QwenQwen2.5-Coder-7B-Instruct-tokenizer` | 445 303 | 3.92 |
| Fryderyk Chopin (PL) | `tokenizer-all-64k` | 13 313 | 4.48 |

## Wnioski

### 1. Vocab size ma znaczenie
Wyraźna zależność liniowa: 8k < 16k < 32k < 64k pod kątem efektywności (mniej tokenów, dłuższe kawałki). Większy słownik = tokenizer uczy się dłuższych podciągów = efektywniejszy embedding. `tokenizer-all-64k` wygrywa na obu PL tekstach.

### 2. Własne tokenizery vs bieliki (teksty PL)
- Własne (64k): **miejsce 1** — 9 434 tokenów vs 13 177 bielik-v3
- Własne (32k all-corpora): miejsce 2–5
- Bielik v3: ~miejsce 4–9 w zależności od tekstu
- Bielik v1/v2: ostatnie miejsce (ten sam tokenizer, różna waga modelu)

Hipoteza z zadania potwierdzona: własne tokenizery trenowane na polskich korpusach efektywniejsze dla PL tekstów.

### 3. Język ma kluczowe znaczenie
Dla angielskiego tekstu (Pickwick Papers) Qwen i bieliki dominują — trenowane na dużych zbiorach z angielskim. Polskie tokenizery radzą sobie słabo z EN (śr.dł. 1.88–2.5 vs 3.92 dla Qwen).

### 4. Specjalizacja vs ogólność
`tokenizer-pan-tadeusz` (trenowany tylko na Panu Tadeuszu) jest 2. najlepszy na tym samym tekście, ale najgorszy na EN i słaby na Chopinie — nadmierna specjalizacja. Tokenizery trenowane na większych, zróżnicowanych korpusach (NKJP, ALL) są bardziej ogólne.

### 5. Bielik v3 lepszy niż v1/v2
Bielik v3 konsekwentnie lepszy od v1/v2 dla PL tekstów (13 177 vs 20 480 tokenów dla Pan Tadeusza). v1 i v2 mają praktycznie identyczny tokenizer.
