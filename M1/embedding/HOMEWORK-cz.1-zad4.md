# Lokalizacje
Tokenizery: ../tokenizer/tokenizers/*.json
Teksty do przemielenia (referencjowane w corpora.py jako CORPORA_DIRS:)
    "NKJP": ../korpus-nkjp/output/*.txt
    "WOLNELEKTURY": ../korpus-wolnelektury/*.txt


# Zadanie 4.1

Folder: `M1/embedding`

Intro:
CBOW (Continuous Bag-of-Words), jest siecią neuronową, która uczy się przewidywać słowo docelowe (środkowe) na podstawie jego słów kontekstowych (otaczających), znajdujących się w określonym oknie.

W pliku `M1/embedding/run-cbow.py` ładuje tokenizer, tokenizuje zadane teksty i buduje w oparciu o nie model embeddingowy typu CBOW (podobieństwo/częstotliwość wystąpień, co w odpowiednio dużej skali zaczyna symulować podobieństwo znaczeniowe - dziwnym trafem tak samo jak LLMy :)) 


Cel zadania: znaleźć takie ustawienia aby słowa pokrewne (np. kobieta-dziewczyna, król-książę były blisko w embeddingu, tj. wartość możliwie bliska 1)

Zadania - w skrócie:
- rozbij skrypt aby dało się osobno trenować i osobno wnioskować (teraz jest wszystko na raz :)
- wybierz słowa/teskt referencyjny (cokolwiek wybierzesz z korpusów lub wymyślisz). Punktem odniesienia mogą być słowa zahardkodowane w skrypcie
- testuj trenowanie na różnych korpusach, rożnych tokenizerach, rożnych parametrach


🔥 Nie wykonasz tego zadania bez skutecznego wykonania poprzedniego zadania. Porównuj pomysły na discordzie.
🔥 Podpowiedź: wsród różnych ustawień, jakie będziesz zmieniać, spróbuj porównać także tokenizery - np. bielik-v1 (mistralowy), bielik-v3 oraz Twój customowy z zadania 3.


Przykładowe oczekiwane wyniki:
```
10 tokenów najbardziej podobnych do SŁOWA 'wojsko' (uśrednione wektory tokenów ['wojsko']):
  > Wektor słowa (początek): [ 0.8360334  0.0980003 -0.221845  -4.700161   1.4552883]...
  - wojsko: 1.0000
  - miasto: 0.6638
  - wojska: 0.6313
  - życie: 0.6036
  - zwycięstwo: 0.5795
  - rycerstwo: 0.5671
  - państwo: 0.5635
  - hetmanów: 0.5599
  - posiłki: 0.5576
  - pospólstwo: 0.5331

10 tokenów najbardziej podobnych do SŁOWA 'szlachta' (uśrednione wektory tokenów ['szlachta']):
  > Wektor słowa (początek): [-1.5282736   0.82800084  1.1820822  -2.4249477   1.0725677 ]...
  - szlachta: 1.0000
  - piechota: 0.6810
  - jazda: 0.6259
  - służba: 0.6035
  - starszyzna: 0.6029
  - wojna: 0.5841
  - armia: 0.5670
  - arystokracja: 0.5617
  - Litwa: 0.5551
  - kupa: 0.5538

10 tokenów najbardziej podobnych do SŁOWA 'choroba' (uśrednione wektory tokenów ['choroba']):
  > Wektor słowa (początek): [ 0.6537147  -0.04082277 -1.689754    0.97554463  0.10579971]...
  - choroba: 1.0000
  - dziewka: 0.6708
  - męka: 0.6616
  - natura: 0.6224
  - taka: 0.6102
  - tęsknota: 0.6064
  - osoba: 0.5888
  - jakaś: 0.5863
  - okrutna: 0.5849
  - zmiana: 0.5797

10 tokenów najbardziej podobnych do SŁOWA 'król' (uśrednione wektory tokenów ['król']):
  > Wektor słowa (początek): [ 2.4375966  -1.0871804  -1.6425471  -1.6709629  -0.62909025]...
  - król: 1.0000
  - książę: 0.7209
  - Chmielnicki: 0.6851
  - mistrz: 0.6605
  - hetman: 0.6238
  - Karol: 0.6195
  - cezar: 0.6113
  - Jurand: 0.5996
  - Bohun: 0.5919
  - jenerał: 0.5895

10 tokenów najbardziej podobnych do kombinacji tokenów: ['dziecko', 'kobieta']
  - dziewczyna: 0.6242
  - ona: 0.6124
  - matka: 0.6069
  - dziewka: 0.6049
  - sztuka: 0.5968
  - piękna: 0.5788
  - męka: 0.5731
  - osoba: 0.5724
  - cnota: 0.5683
  - sama: 0.5590
```

król - książe - 0.7209 jest niezłe (choć mogłoby być lepsze)
['dziecko', 'kobieta'] - dziewczyna: 0.6242 - też niezłe i też mogłoby być lepsze.

Oczywiście śmiało podmieniaj słowa.

# Zadanie 4.2

Szukamy najbardziej podobnych **zdań**: `M1/embedding/run-doc2vec.py`

Trenujemy nasz własny model embedingowy (dla całych zdań, nie samych słów).

Zadanie - j/w - zoptymalizuj trening.

Dobierz parametry treningu (analogicznie co wcześniej) tak, aby optymalnie: zwiększyć jakość wychwytywania podobieństwa i jednocześnie wykonywać trening najkrócej (tj. nie tracić czasu i/lub nie "przetrenować" modelu)

W razie potrzeby dostosowuj korpus treningowy.

Zmiana których parametrów najbardziej wydłuża trening?

Przykładowy output (całkiem sensowny):
```
Zdanie do wnioskowania: "Jestem głodny i bardzo chętnie zjadłbym coś."
 najbardziej podobnych zdań z korpusu:
  - Sim: 0.6533 | Zdanie: Po chwili otworzył je. Rzędzian siedział ciągle pod oknem.
  - Sim: 0.6525 | Zdanie: – ja też nie jem
  - Sim: 0.6487 | Zdanie: — Nie bój się waćpanna, nie zjem cię!
  - Sim: 0.6338 | Zdanie: – ja też nie jem znaczy pewno zjem ale nie nie lubię ale jest podobno taka zdrowa..
  - Sim: 0.6330 | Zdanie: — Przysłał im je król francuski — odrzekł opat.
```
Przykładowy output (zdecydowanie bezsensowny):
```
Zdanie do wnioskowania: "Jestem głodny i bardzo chętnie zjadłbym coś."
5 najbardziej podobnych zdań z korpusu:
  - Sim: 0.9350 | Zdanie: – prawdopodobnie. nie nie wiesz po prostu
  - Sim: 0.9336 | Zdanie: – a teraz. ktoś mądrze powiedział że..
  - Sim: 0.9332 | Zdanie: Dość, gdy niebezpieczeństwa i śmiałość przypomnę,
  - Sim: 0.9250 | Zdanie: Jestem twój stryj; choć stary, znam, co serce młode;
  - Sim: 0.9207 | Zdanie: Poleciałabym ja
```

# Zadanie 4.3

Plik `M1/embedding/run-sbert.py` - korzystamy z wcześniej wytrenowanego modelu o który się opieramy. Bierzemy nasze zdania i kodujemy je w "bazie danych" wektorowej (macierz embeddingów zdań z naszego korpusu). I (zwyczajnie) odpytujemy tę bazę w odniesieniu do zadanego zdania (które wcześniej trzeba zaembedować).

Poszukiwanie najbliższego wektora w wielowymiarowej przestrzeni.

**Skrypt realizuje swoje zadanie**.

Twoja rola:
- spróbuj znaleźć alternatywę dla modelu `SentenceTransformer(MODEL_NAME)` lepiej dostosowanego do języka polskiego
- rozdziel skrypt tak, aby "kodowanie bazy danych" było osobno a sprawdzanie zbieźności osobno.
- odpytaj o zdania wymyślone oraz takie które bezpośrednio pochodzą z korpusu treningowego.
