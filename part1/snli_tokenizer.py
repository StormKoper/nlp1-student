## This file handles the BPE as per the tutorial
from datasets import load_from_disk

from experiment import train_tokenizer
from snli_config import VOCAB_SIZE, MIN_FREQUENCY, DATA_PATH, TOKENIZER_PATH


def main() -> None:
    snli = load_from_disk(DATA_PATH)

    # Specifying the corpus
    def corpus():
        for example in snli["train"]:
            yield example["premise"]
            yield example["hypothesis"]

    # We steal the BPE from the tutorial 
    bpe_tokenizer = train_tokenizer(
        corpus(),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY
    )

    TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
    bpe_tokenizer.save(str(TOKENIZER_PATH / "bpe.json"))

if __name__ == "__main__":
    main()
