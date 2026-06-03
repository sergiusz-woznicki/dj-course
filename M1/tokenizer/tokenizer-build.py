import argparse

from corpora import CORPORA_FILES
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

OUTPUT_NAMES = {
    "PAN_TADEUSZ": "tokenizers/tokenizer-pan-tadeusz.json",
    "WOLNELEKTURY": "tokenizers/tokenizer-wolnelektury.json",
    "NKJP": "tokenizers/tokenizer-nkjp.json",
    "ALL": "tokenizers/tokenizer-all-corpora.json",
}

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", required=True, choices=list(CORPORA_FILES.keys()), help="Korpus treningowy")
parser.add_argument("--output", help="Ścieżka wyjściowa (opcjonalna)")
parser.add_argument("--vocab-size", type=int, default=32000, help="Rozmiar słownika (default: 32000)")
args = parser.parse_args()

FILES = [str(f) for f in CORPORA_FILES[args.corpus]]
OUTPUT_FILE = args.output or OUTPUT_NAMES[args.corpus]


print(f"Korpus: {args.corpus}")
print(f"Liczba plików: {len(FILES)}")
files_preview = str(FILES)
print(f"Pliki: {files_preview[:100]}...")
print(f"Output: {OUTPUT_FILE}")

# 1. Initialize the Tokenizer (BPE model)
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 2. Set the pre-tokenizer (e.g., split on spaces)
tokenizer.pre_tokenizer = Whitespace()

# 3. Set the Trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=args.vocab_size,
    min_frequency=2,
)

# 4. Train the Tokenizer
tokenizer.train(FILES, trainer=trainer)

# 5. Save the vocabulary and tokenization rules
tokenizer.save(OUTPUT_FILE)

for txt in [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
]:
    encoded = tokenizer.encode(txt)
    print("Zakodowany tekst:", encoded.tokens)
    print("ID tokenów:", encoded.ids)
