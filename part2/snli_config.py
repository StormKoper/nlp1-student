from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = ROOT / "data" / "snli"
EMBEDDING_PATH: Path = ROOT / "data" / "embedding"
VOCABULARY_PATH: Path = ROOT / "data" / "vocabulary"
TOKENIZATION_PATH: Path = ROOT / "data" / "tokenization"
EMB_AND_POOL_PATH: Path = ROOT / "data" / "emb_and_pool"

# Folding and Seed
K: int = 5
SEED: int = 42

# Embeddings
PROVIDER: str = "fasttext"
EMBEDDING: str = "crawl-300d-2M-subword.zip"
EMB_AND_POOL_MODELS = "embed and pool models"
DIMS: int = 300
SKIP_FIRST: bool = True
