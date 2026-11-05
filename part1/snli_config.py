from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "snli"
TOKENIZER_PATH = ROOT / "data" / "tokenizers"
REPRESENTATIONS_PATH = ROOT / "data" / "representations"

# Folding and Seed
K = 5
SEED = 42

# Tokenization 
VOCAB_SIZE = 10000
MIN_FREQUENCY = 5
