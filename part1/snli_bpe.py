## This file handles the BPE as per the tutorial
from pathlib import Path
from datasets import load_from_disk
from part1.experiment import train_tokenizer

# Hyperparameters, guessed roughly for now
VOCAB_SIZE = 10000
MIN_FREQUENCY = 5

# Loading the file
ROOT = Path(__file__).resolve().parent
snli = load_from_disk(ROOT / "data" / "snli")

# Specifying the corpus 
def corpus():
    for example in snli["train"]:
        yield example["premise"]
        yield example["hypothesis"]

# We steal the BPE from the tutorial 
tokenizer = train_tokenizer(
    corpus(),
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY
)

tokenizer.save(str(ROOT / "snli_bpe.json"))        