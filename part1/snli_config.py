from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = ROOT / "data" / "snli"
TOKENIZER_PATH: Path = ROOT / "data" / "tokenizers"
REPRESENTATIONS_PATH: Path = ROOT / "data" / "representations"
ARCHITECTURES_PATH: Path = ROOT / "data" / "architectures"
PREDICTIONS_PATH: Path = ROOT / "data" / "predictions"

# Folding and Seed
K: int = 5
SEED: int = 42

# Tokenization 
VOCAB_SIZE: int = 10000
MIN_FREQUENCY: int = 5

# Representation
TOKENIZERS: list = ["snli", "mistral"]
