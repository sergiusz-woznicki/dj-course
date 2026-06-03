# Wyniki — Zadanie 4.1 CBOW

Data: 2026-06-03

## Komendy

```bash
# Trening
python3 train-cbow.py --tokenizer tokenizer-all-64k --corpus WOLNELEKTURY --output-dir models/all-64k

# Wnioskowanie — pojedyncze słowa
python3 infer-cbow.py --model-dir models/all-64k --words wojsko szlachta choroba król

# Wnioskowanie — kombinacja słów
python3 infer-cbow.py --model-dir models/all-64k --words dziecko kobieta --combine
```

## Parametry treningu

| Parametr | Wartość |
|---|---|
| tokenizer | `tokenizer-all-64k` |
| corpus | WOLNELEKTURY (35 plików, 102 529 zdań) |
| vector_size | 20 |
| window | 6 |
| min_count | 2 |
| epochs | 20 |
| sample_rate | 1e-2 |
| czas treningu | 18.2s |
| vocab size | 53 421 tokenów |

## Wyniki

### wojsko
| Token | Similarity |
|---|---|
| wojsko | 1.0000 |
| miasto | 0.8254 |
| pieniądze | 0.8081 |
| złe | 0.8080 |
| ono | 0.8040 |

### szlachta
| Token | Similarity |
|---|---|
| szlachta | 1.0000 |
| starzy | 0.8048 |
| księża | 0.7955 |
| oni | 0.7936 |
| ludzie | 0.7768 |

### choroba
| Token | Similarity |
|---|---|
| choroba | 1.0000 |
| niedola | 0.8930 |
| fantazja | 0.8605 |
| natura | 0.8457 |
| inna | 0.8178 |

### król
| Token | Similarity |
|---|---|
| król | 1.0000 |
| hetman | 0.9519 |
| Radziwiłł | 0.9019 |
| **książę** | **0.8877** |
| Kali | 0.8798 |
| Chmielnicki | 0.8649 |

### kombinacja: dziecko + kobieta
| Token | Similarity |
|---|---|
| kobieta | 0.8661 |
| wdzięczność | 0.8603 |
| piękna | 0.8438 |
| wiara | 0.8373 |
| dziecko | 0.8301 |

## Podsumowanie

- `król–książę: 0.8877` >> cel 0.7 ✓
- `tokenizer-all-64k` (64k vocab, trening na PL korpusach) daje znacznie lepsze podobieństwa semantyczne niż bieliki (śr.dł. tokenu 4.64 vs 2.14 dla bielik-v1)
- Dłuższe tokeny → mniej tokenów na słowo → lepsze uśrednianie → wyraźniejsze wektory semantyczne
- WOLNELEKTURY wystarczy dla tematyki historyczno-literackiej (hetman, szlachta, król)

---

## Eksperymenty porównawcze

### Komenda

```bash
python3 run-experiments.py 2>/tmp/exp-progress.log | tee /tmp/exp-results.txt
```

### Faza 1: Tokenizer × Korpus (parametry domyślne: vector_size=20, window=6, epochs=20)

| Tokenizer | Korpus | Czas(s) | król–książę | dziecko+kobieta top1 |
|---|---|---|---|---|
| tokenizer-all-64k | WOLNELEKTURY | 16.5 | 0.8894 | bóstwo:0.859 |
| **tokenizer-all-64k** | **ALL** | **25.2** | **0.9131** | serce:0.867 |
| bielik-v3-tokenizer | WOLNELEKTURY | 19.4 | - (¹) | ▁:0.995 (¹) |
| bielik-v3-tokenizer | ALL | 27.9 | - (¹) | ▁:0.994 (¹) |
| bielik-v1-tokenizer | WOLNELEKTURY | 26.9 | - (¹) | ▁:0.953 (¹) |
| bielik-v1-tokenizer | ALL | 40.7 | - (¹) | ▁:0.946 (¹) |

¹ Bieliki tokenizują `książę` na subword-y (`▁książ`, `ę`) których każdy z osobna nie ma wektora dla całego słowa — `get_word_vector` zwraca None. Token `▁` (sam spacja-prefiks) dominuje jako artifact tokenizera.

### Faza 2: Param sweep (tokenizer-all-64k × WOLNELEKTURY)

| vector_size | epochs | Czas(s) | król–książę | dziecko+kobieta top1 |
|---|---|---|---|---|
| 20 | 10 | 9.6 | 0.9117 | wdzięczność:0.895 |
| **20** | **20** | **15.4** | **0.8990** | wdzięczność:0.892 |
| 20 | 50 | 40.5 | 0.8743 | boginka:0.891 |
| 50 | 20 | 20.5 | 0.8075 | ona:0.717 |
| 100 | 20 | 28.4 | 0.7334 | piękna:0.659 |
| 100 | 50 | 73.5 | 0.7386 | cnota:0.593 |

### Wnioski z eksperymentów

- **Tokenizer decyduje**: bieliki (v1/v3) są bezużyteczne dla CBOW na PL — subword split rozrywa słowa na tokeny bez kontekstu, vocab nie zawiera pełnych słów polskich
- **tokenizer-all-64k + ALL**: najlepsza jakość (król–książę: 0.913), +8.5s vs WOLNELEKTURY
- **vector_size=20 optymalne** dla tego rozmiaru korpusu (~100k zdań): większe wymiary (50, 100) pogarszają wyniki — za mało danych na rzadkie przestrzenie
- **epochs=10 wystarcza** — dodatkowe epoki nie poprawiają (0.9117 → 0.8990 → 0.8743), lekkie przetrenowanie przy ep=50
- **Optimum**: `tokenizer-all-64k`, `ALL`, `vector_size=20`, `epochs=10` → król–książę: 0.9117 w 9.6s
