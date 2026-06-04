# Wyniki: Zadanie 4.3 — SBERT split encode/infer + lepszy model PL

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Korpus | PAN_TADEUSZ (10 061 zdań) |
| Model PL | `sdadas/st-polish-paraphrase-from-mpnet` |
| Model baseline | `intfloat/multilingual-e5-small` |
| Sprzęt | CPU (brak GPU) |

## Pliki

- `encode-sbert.py` — enkodowanie korpusu → `sentence_embeddings.npy` + `sentences.json` + `meta.json`
- `infer-sbert.py` — odpytywanie bazy na podstawie zdania wejściowego

## Czasy enkodowania

| Model | Zdania | Dim | Czas |
|---|---|---|---|
| `st-polish-paraphrase-from-mpnet` | 10 061 | 768 | 198.8s |
| `multilingual-e5-small` | 10 061 | 384 | 34.6s |

## Wyniki — model PL (`st-polish-paraphrase-from-mpnet`)

### Zapytanie: "Jestem głodny i bardzo chętnie zjadłbym coś."
```
  - Sim: 0.7885 | Zdanie: Studzi gniewy, zapala potrzebę jedzenia.
  - Sim: 0.7333 | Zdanie: Ale my głodni, każ Wać przynosić potrawy".
  - Sim: 0.7005 | Zdanie: A mająca potrawkę z sosem u ogona.
  - Sim: 0.6981 | Zdanie: Jaki się przy obiadach i wieczerzach chowa.
  - Sim: 0.6879 | Zdanie: Zostawię mu kęs niezły szlacheckiego chleba;
```

### Zapytanie (zdanie z korpusu): "Studzi gniewy, zapala potrzebę jedzenia."
```
  - Sim: 1.0000 | Zdanie: Studzi gniewy, zapala potrzebę jedzenia.   ← idealne
  - Sim: 0.8258 | Zdanie: Wzbudza oskomę w ustach, głód w żołądkach rodzi.
  - Sim: 0.8211 | Zdanie: Ale my głodni, każ Wać przynosić potrawy".
  - Sim: 0.8037 | Zdanie: Jaki się przy obiadach i wieczerzach chowa.
  - Sim: 0.7991 | Zdanie: Idzie, z biedy i z głodu przymierając czasem,
```

### Zapytanie: "Wojsko wejdzie do miast i skończą się bunty."
```
  - Sim: 0.7692 | Zdanie: Takt marszu, wojna, atak, szturm, słychać wystrzały,
  - Sim: 0.7434 | Zdanie: Ja mówię, będzie wojna u nas. Do majora
  - Sim: 0.7301 | Zdanie: Wieśniacy i żołnierstwo ścisnęło się kołem.
  - Sim: 0.7292 | Zdanie: Podług ustaw wojennych za takową psotę
  - Sim: 0.7275 | Zdanie: Co mówię! wszak Polacy miewali zamieszki
```

## Wyniki — baseline (`multilingual-e5-small`)

### Zapytanie: "Jestem głodny i bardzo chętnie zjadłbym coś."
```
  - Sim: 0.9028 | Zdanie: Jestem krewną, jedyną Zosi opiekunką.
  - Sim: 0.8967 | Zdanie: I kuchcik, małe, ale bardzo silne chłopię.
  - Sim: 0.8954 | Zdanie: Że go przyjmuję grzecznie, chce mnie za nos wodzić.
  - Sim: 0.8899 | Zdanie: Abyś został; wnet skończę, ledwie mam dość mocy
  - Sim: 0.8883 | Zdanie: Szlachta głodna plądruje, zabiera, co może.
```

## Wnioski

### 1. Model dedykowany PL >> multilingual
`st-polish-paraphrase-from-mpnet` jest trenowany na polskich parach parafraz → rozumie semantykę PL.
Mimo niższych wartości sim (0.79 vs 0.90), trafność tematyczna jest zdecydowanie lepsza.

`multilingual-e5-small` zwraca zdania z podobną strukturą syntaktyczną ("Jestem..."), nie semantyką.
Wysoka wartość sim (0.90+) jest myląca — to "cosine inflation", nie prawdziwe podobieństwo.

### 2. Zdanie z korpusu → sim = 1.0
Enkodowanie deterministyczne: to samo zdanie → identyczny wektor → sim=1.0. Potwierdza poprawność pipeline.

### 3. Podział encode/infer — sens praktyczny
- Enkodowanie 10k zdań: 199s (mpnet) / 35s (e5). Jeden raz, zapisane na dysk.
- Wnioskowanie: <2s (ładowanie pliku + encode 1 zdania).
- Przy dużych korpusach (100k+ zdań) oszczędność ogromna — baza wektorowa nie wymaga przebudowy.

### 4. Rozmiar modelu a czas
mpnet (dim=768) — 5.7× wolniejszy od e5-small (dim=384), dim 2× większy.
Na GPU różnica byłaby minimalna (paralelizacja batch). Na CPU szybkość ∝ 1/dim.

### 5. Alternatywne modele PL warte przetestowania
- `sdadas/st-polish-paraphrase-from-distilroberta` — szybszy, zbliżona jakość
- `Voicelab/slt-roberta-large-polish` — większy, lepsza jakość ale wolniejszy
- `allegro/herbert-large-cased` — wymaga wrapper (nie jest sentence-transformer out of box)

## Komendy

```bash
# enkodowanie bazy
python encode-sbert.py --corpus PAN_TADEUSZ --model sdadas/st-polish-paraphrase-from-mpnet --output-dir models/sbert-pl

# wnioskowanie
python infer-sbert.py --model-dir models/sbert-pl --sentence "Jestem głodny i bardzo chętnie zjadłbym coś." --topn 5

# baseline
python encode-sbert.py --corpus PAN_TADEUSZ --model intfloat/multilingual-e5-small --output-dir models/sbert-e5
python infer-sbert.py --model-dir models/sbert-e5 --sentence "Jestem głodny i bardzo chętnie zjadłbym coś." --topn 5
```
