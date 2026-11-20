from itertools import zip_longest

from datasets import DatasetDict, load_from_disk
from sacremoses import MosesTokenizer

from . import clslib
from .snli_config import (
    DATA_PATH,
    DIMS,
    EMBEDDING,
    EMBEDDING_PATH,
    SKIP_FIRST,
    TOKENIZATION_PATH,
)


class SNLIPairTokenizer(clslib.Tokenizer):
    """Convert each sentence pair to two columns of word IDs for the GRU."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.pad_id = embedding_model.vocab.index(embedding_model.pad_token)
        self.tokenizer = MosesTokenizer(lang="en")

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence, escape=False)

    def tokens(self, pair):
        return self.tokenize(pair[0]), self.tokenize(pair[1])

    def ids(self, pair):
        premise, hypothesis = self.tokens(pair)
        unk_id = self.embedding_model.vocab.default_index
        premise_ids = [self.embedding_model.index(word) for word in premise] or [unk_id]
        hypothesis_ids = [self.embedding_model.index(word) for word in hypothesis] or [unk_id]
        return list(zip_longest(premise_ids, hypothesis_ids, fillvalue=self.pad_id))

# running the tokenizer and saving to cache
def main() -> None:
    embedding_file = EMBEDDING_PATH / EMBEDDING

    # check that cached things exist 
    if not DATA_PATH.is_dir() or not embedding_file.is_file():
        raise FileNotFoundError(
            "Cached SNLI data or embeddings are missing. "
            "Run python -m part2.snli_processing first."
        )

    # load cached dataset and fasttext embeddings
    snli = load_from_disk(DATA_PATH)
    embedding_model = clslib.PretrainedEmbeddings(
        str(embedding_file), DIMS, skip_first_line=SKIP_FIRST
    )
    tokenizer = SNLIPairTokenizer(embedding_model)

    # save ids, labels and existing folds for every split. raw text stays in DATA_PATH.
    tokenized_splits = {}

    for split, data in snli.items():

        # turning tokenizer output into dictionary entries
        def convert_example(example):
            premise = example["premise"]
            hypothesis = example["hypothesis"]

            ids = tokenizer.ids((premise, hypothesis))

            return {"input_ids": ids}

        # changes fields from ['premise', 'hypothesis', 'label', 'fold'] to ['label', 'fold', 'input_ids']
        tokenized_data = data.map(
            convert_example,
            remove_columns=["premise", "hypothesis"],
            load_from_cache_file=False,
            new_fingerprint=f"snli-{split}-tokenized",
            desc=f"Tokenizing SNLI {split}",
        )

        tokenized_splits[split] = tokenized_data

    tokenized = DatasetDict(tokenized_splits)
    tokenized.save_to_disk(TOKENIZATION_PATH)
    print(f"Saved tokenized SNLI to {TOKENIZATION_PATH}")


if __name__ == "__main__":
    main()
