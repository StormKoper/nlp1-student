## This file handles the BPE as per the tutorial
from pathlib import Path
from datasets import load_from_disk
from part1.experiment import train_tokenizer

from .snli_config import VOCAB_SIZE, MIN_FREQUENCY, DATA_PATH, TOKENIZER_PATH


snli = load_from_disk(DATA_PATH)

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

tokenizer.save(str(TOKENIZER_PATH))        