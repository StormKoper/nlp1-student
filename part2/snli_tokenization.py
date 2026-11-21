from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
from datasets import DatasetDict, load_from_disk
from sacremoses import MosesTokenizer

from . import clslib
from .snli_config import (
    DATA_PATH,
    DIMS,
    EMBEDDING,
    EMBEDDING_PATH,
    FIGURES_PATH,
    SKIP_FIRST,
    TOKENIZATION_PATH,
)


class SNLIPairTokenizer(clslib.Tokenizer):
    """Convert each sentence pair to two columns of word IDs for the GRU."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.unk_id = embedding_model.vocab.default_index
        self.pad_id = embedding_model.vocab.index(embedding_model.pad_token)
        self.tokenizer = MosesTokenizer(lang="en")

    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence, escape=False)

    def tokens(self, pair):
        return self.tokenize(pair[0]), self.tokenize(pair[1])

    def ids(self, pair):
        premise, hypothesis = self.tokens(pair)
        premise_ids = [self.embedding_model.index(word) for word in premise] or [self.unk_id]
        hypothesis_ids = [self.embedding_model.index(word) for word in hypothesis] or [self.unk_id]
        return premise_ids, hypothesis_ids


def plot_length_histograms(lengths, output_path="token_lengths.png"):
  splits = list(lengths.keys())
  fig, axes = plt.subplots(
      len(splits), 2, figsize=(10, 3 * len(splits)),
      squeeze=False, constrained_layout=True,
  )

  for row, split in enumerate(splits):
    prem_lens = np.array(lengths[split]["premise"])
    hyp_lens = np.array(lengths[split]["hypothesis"])

    for col, (arr, label) in enumerate(
        [(prem_lens, "Premise"), (hyp_lens, "Hypothesis")],
    ):
      ax = axes[row, col]
      bins = np.arange(0.5, arr.max() + 1.5, 1) if len(arr) else 10
      ax.hist(arr, bins=bins, edgecolor="black", alpha=0.8)
      ax.set_title(
          f"{split.capitalize()} — {label} token count (n={len(arr):,})"
      )
      ax.set_xlabel("Number of tokens")
      ax.set_ylabel("Examples")
      if len(arr):
        ax.axvline(
            arr.mean(),
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={arr.mean():.1f}",
        )
        ax.legend()

  plt.savefig(FIGURES_PATH / output_path, dpi=150)
  plt.close(fig)


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

    totals = {}
    unks = {}
    lengths = {}
    for split, data in snli.items():
        split_total = split_unk = 0
        premise_lens = []
        hypothesis_lens = []

        # turning tokenizer output into dictionary entries
        def convert_example(example):
            premise = example["premise"]
            hypothesis = example["hypothesis"]

            premise_ids, hypothesis_ids = tokenizer.ids((premise, hypothesis))
            paired_ids = list(zip_longest(premise_ids, hypothesis_ids, fillvalue=tokenizer.pad_id))

            total = premise_ids + hypothesis_ids
            unk = sum(i == tokenizer.unk_id for i in total)
            nonlocal split_total, split_unk
            split_total += len(total)
            split_unk += unk
            premise_lens.append(len(premise_ids))
            hypothesis_lens.append(len(hypothesis_ids))

            return {"input_ids": paired_ids}

        # changes fields from ['premise', 'hypothesis', 'label', 'fold'] to ['label', 'fold', 'input_ids']
        tokenized_data = data.map(
            convert_example,
            remove_columns=["premise", "hypothesis"],
            load_from_cache_file=False,
            new_fingerprint=f"snli-{split}-tokenized",
            desc=f"Tokenizing SNLI {split}",
        )

        tokenized_splits[split] = tokenized_data

        totals[split] = split_total
        unks[split] = split_unk
        lengths[split] = {
            "premise": premise_lens,
            "hypothesis": hypothesis_lens,
        }

    tokenized = DatasetDict(tokenized_splits)
    tokenized.save_to_disk(TOKENIZATION_PATH)
    print(f"Saved tokenized SNLI to {TOKENIZATION_PATH}")

    for split, total in totals.items():
        print(f"{str(split).capitalize()}: coverage {total-unks[split]:,} / {total:,} ({(total-unks[split])/total:%})")

    plot_length_histograms(lengths)
    print("Saved histogram figure")

if __name__ == "__main__":
    main()
