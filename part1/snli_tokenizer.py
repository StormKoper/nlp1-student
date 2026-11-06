## This file handles the BPE as per the tutorial
from datasets import load_from_disk
from experiment import train_tokenizer
from snli_config import DATA_PATH, MIN_FREQUENCY, TOKENIZER_PATH, VOCAB_SIZE
from tokenizers import Tokenizer


def main() -> None:
    snli = load_from_disk(DATA_PATH)

    # Specifying the corpus
    def corpus():
        for example in snli["train"]:
            yield example["premise"]
            yield example["hypothesis"]

    # We steal the BPE from the tutorial 
    snli_tokenizer = train_tokenizer(
        corpus(),
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY
    )

    TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
    snli_tokenizer.save(str(TOKENIZER_PATH / "snli_bpe.json"))

    pretrained_tokenizer = Tokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    pretrained_tokenizer.save(str(TOKENIZER_PATH / "mistral_bpe.json"))

if __name__ == "__main__":
    main()
