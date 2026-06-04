# Wyniki: Zadanie 4.2 - Doc2Vec

## Konfiguracja (d2v-v2)

| Parametr | Wartość |
|---|---|
| Corpus | WOLNELEKTURY (35 plików) |
| Tokenizer | bielik-v3-tokenizer |
| vector_size | 100 |
| window | 8 |
| min_count | 4 |
| epochs | 20 |
| dm | 1 (PV-DM) |
| min_sentence_len | 5 słów |
| Zdań po filtrze | 90 471 |
| Czas treningu | 315.6s |

## Wyniki wnioskowania

### "Jestem głodny i bardzo chętnie zjadłbym coś."
```
  - Sim: 0.8167 | Zaprawdę, i rok cały przesiedziałbym chętnie
  - Sim: 0.8095 | i własną duszę poznać nieskończoną,
  - Sim: 0.8031 | — Wiem coś o tym — dodał Zagłoba.
  - Sim: 0.8022 | Pierwszy raz posłyszała w życiu z ust młodziana
  - Sim: 0.8014 | W tej chwili zbliżył się do niego Klejn z jakimś listem.
```

### "Rzędzian siedział ciągle pod oknem."
```
  - Sim: 0.8942 | — Pójdziemy! — odrzekł Ursus.
  - Sim: 0.8874 | — Nie chodź! Dobrze z tobą…
  - Sim: 0.8856 | Pani baronowa zaczęła się niepokoić.
  - Sim: 0.8833 | Następnie zwrócił się do Wrzeszczowicza.
  - Sim: 0.8824 | Rzecki pochylił się ku niemu i otworzył usta.
```

### "Bitwa była krwawa i długa."
```
  - Sim: 0.8950 | — Pójdziemy! — odrzekł Ursus.
  - Sim: 0.8852 | Pani baronowa zaczęła się niepokoić.
  - Sim: 0.8824 | Następnie zwrócił się do Wrzeszczowicza.
  - Sim: 0.8819 | — Nie będąc nigdy niewolnicą, nie mogła być wyzwolona.
  - Sim: 0.8802 | Po czym wyprostował się i czekał.
```

## Wnioski

### Co działa
- Filtr `--min-sentence-len 5` eliminuje szum (nazwy mówców z dramatów: "FAUST", "MEFISTOFELES")
- "Jestem głodny... chętnie zjadłbym" → "przesiedziałbym chętnie" — wspólne słowo + kontekst pożądania ✓
- Brak nonsensownych wyników z sim ~0.93 (jak w przykładzie "złego" outputu z zadania)

### Obserwacje
- Sim 0.80–0.89 dla wszystkich wyników — zakres nieco zawężony, model nie jest bardzo dyskryminatywny
- Zdania krótkie i narracyjne ("— Pójdziemy!") pojawiają się dla różnych zapytań → wektory krótkich zdań zbiegają się do wspólnego obszaru
- PV-DM (dm=1) z window=8 działa poprawnie dla prozy narracyjnej

### Co najbardziej wydłuża trening
- `vector_size` — wzrost liniowy z rozmiarem wektora
- `epochs` — wzrost liniowy (20 epok = 315s; 40 epok ≈ 630s)
- `corpus` — ALL zamiast WOLNELEKTURY ≈ 2–3× więcej zdań → 2–3× dłużej

### Do eksperymentowania
- więcej epok (40+) na dużym korpusie — lepsza dyskryminacja
- porównanie tokenizerów: bielik-v1 vs bielik-v3 vs customowy

## Pliki modelu
- `models/d2v-v2/doc2vec.model`
- `models/d2v-v2/sentence_map.json`
- `models/d2v-v2/meta.json`

---

# Eksperyment 2: PV-DBOW (dm=0)

## Konfiguracja (d2v-dbow)

| Parametr | Wartość |
|---|---|
| Corpus | WOLNELEKTURY (35 plików) |
| Tokenizer | bielik-v3-tokenizer |
| vector_size | 100 |
| window | 8 |
| min_count | 4 |
| epochs | 20 |
| dm | 0 (PV-DBOW) |
| min_sentence_len | 5 słów |
| Zdań | 90 471 |
| **Czas treningu** | **101.0s** (vs 315.6s DM — 3× szybszy) |

## Wyniki wnioskowania

### "Jestem głodny i bardzo chętnie zjadłbym coś."
```
  - Sim: 0.6613 | Choćbym pięć, sześć lat duszkiem opowiadał biedy
  - Sim: 0.6595 | i dobry człowiek, i serdeczny,
  - Sim: 0.6548 | lubię porządek i czyste sumienie,
  - Sim: 0.6496 | Wolałbym jednak być tam w górze,
  - Sim: 0.6495 | — No, mój panie, wolę być trochę szpakowatym aniżeli łysym — oburzył się pan Ignacy.
```

### "Rzędzian siedział ciągle pod oknem."
```
  - Sim: 0.8582 | Po chwili otworzył je. Rzędzian siedział ciągle pod oknem.  ← DOKŁADNE TRAFIENIE ✓
  - Sim: 0.7916 | Rzędzian usiadł w nogach łóżka…
  - Sim: 0.7877 | Rzędzian zaczął się drapać w głowę.
  - Sim: 0.7664 | Pan Wołodyjowski spojrzał i wykrzyknął:
  - Sim: 0.7650 | Nagle jakiś głos przeraźliwy tuż pod domem począł krzyczeć:
```

### "Bitwa była krwawa i długa."
```
  - Sim: 0.7852 | Coraz częstsze błyskawice rozświecały okna; burza była bliżej i bliżej.
  - Sim: 0.7726 | Bitwa miała tuż, tuż nastąpić.  ← semantycznie trafne ✓
  - Sim: 0.7671 | niż ta wśród gór i łóz.
  - Sim: 0.7623 | Mocna tratwa wytrzyma bijące bałwany,
  - Sim: 0.7589 | Wszystkie włókna trzeszczały, skwarząc się w źrenicy.
```

## Wnioski DBOW vs DM

| | DM (dm=1) | DBOW (dm=0) |
|---|---|---|
| Czas treningu | 315.6s | **101.0s** |
| Zakres sim | 0.80–0.89 | 0.65–0.86 |
| Dyskryminacja | słaba (wąski zakres) | lepsza (szerszy zakres) |
| Dokładne trafienie zdania z korpusu | nie | **tak** (sim 0.86) |
| Jakość semantyczna | przeciętna | **lepsza** |

**Wniosek: DBOW (dm=0) jest lepszym wyborem** — 3× szybszy trening i wyraźnie lepsza jakość semantyczna. PV-DBOW ignoruje kolejność słów i uczy się reprezentacji dokumentu niezależnie od kontekstu okna, co dla całych zdań działa lepiej niż PV-DM.
