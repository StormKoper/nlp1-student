from os import replace
from shutil import copyfileobj
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from clslib import PretrainedEmbeddings
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from sklearn.model_selection import StratifiedKFold
from snli_config import (
    DATA_PATH,
    DIMS,
    EMBEDDING,
    EMBEDDING_PATH,
    PROVIDER,
    SEED,
    SKIP_FIRST,
    K,
)


# Helper function to perform quality control on the data
def valid(example):
    return (
        example["label"] in (0, 1, 2)
        and isinstance(example["premise"], str)
        and isinstance(example["hypothesis"], str)
        and example["premise"].strip()
        and example["hypothesis"].strip()
    )

# Helper function to assign folds to data
# Used for cross-validation in the training set
def assign_folds(data):
    fold_ids = np.empty(len(data), dtype=int)
    
    kf = StratifiedKFold(
        n_splits=K, 
        shuffle=True, 
        random_state=SEED
    ) 

    # Assiging an element the id of the fold in which it is in the validation set
    for fold, (_, val_indices) in enumerate(kf.split(data, data["label"])):
        fold_ids[val_indices] = fold

    # Reconfigure the training dataset to have these k-fold ids
    # The test set will be used for testing, and the validation set will be unused
    return data.add_column("cv_fold", fold_ids.tolist()), fold_ids


def get_snli_dataset():
    # Check whether the dataset already exists
    if DATA_PATH.exists():
        snli = load_from_disk(DATA_PATH)
        fold_ids = np.array(snli["train"]["cv_fold"])
        return snli, fold_ids

    # 1. Download SNLI from Hugging Face
    snli = load_dataset("stanfordnlp/snli")

    # 2. Prune bad examples with invalid labels or empty inputs
    snli = DatasetDict({
        split: data.filter(valid)
        for split, data in snli.items()
    })

    # 3. Assign each training example to one of K folds
    snli["train"], fold_ids = assign_folds(snli["train"])

    # 4. Merge the validation set into the test set
    snli["test"] = concatenate_datasets([snli["validation"], snli["test"]])
    del snli["validation"]

    # 5. Save the processed dataset locally
    snli.save_to_disk(DATA_PATH)
    print(f"\nSaved dataset to {DATA_PATH}")

    return snli, fold_ids


def print_snli_summary(snli, fold_ids):
    label_meanings = ["entailment", "neutral", "contradiction"]

    for split, data in snli.items():
        labels = np.array(data["label"])
        count = len(data)
        print(f"\nThe {split} split has {count:,} examples with labels:")
        for label in np.unique(labels):
            label_count = np.sum(labels == label)
            print(f"\tLabel {label} ({label_meanings[label]}): {label_count:,} ({100*label_count/count:.2f}%) examples")

    train_labels = np.array(snli["train"]["label"])
    for fold in range(K):
        mask = fold_ids == fold
        labels = train_labels[mask]
        count = mask.sum()
        print(f"\nTraining fold {fold} has {count:,} examples with labels:")
        for label in np.unique(labels):
            label_count = np.sum(labels == label)
            print(f"\tLabel {label} ({label_meanings[label]}): {label_count:,} ({100*label_count/count:.2f}%) examples")


# Helper function to make the downloaded embedding model compatible with the provided code
def isolate_embedding_txt():
    path = EMBEDDING_PATH / EMBEDDING
    temp_path = path.with_suffix(".tmp")
    with ZipFile(path, "r") as read_zip, ZipFile(temp_path, "w") as write_zip:
        for item in read_zip.infolist():
            if item.filename == path.with_suffix(".vec").name:
                with read_zip.open(item.filename, "r") as read_file, \
                     write_zip.open(path.with_suffix(".txt").name, "w", force_zip64=True) as write_file:
                    copyfileobj(read_file, write_file, length=1024 * 1024)
    replace(temp_path, path)


def get_embedding_model():
    if PROVIDER == "fasttext":
        path = EMBEDDING_PATH / EMBEDDING
        if not path.is_file():
            EMBEDDING_PATH.mkdir(parents=True, exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/"
            print("Downloading the embedding model...")
            urlretrieve(url + EMBEDDING, path)
            print("Converting downloaded model to .txt...")
            isolate_embedding_txt()
        print("Setting up the embedding model...")
        embedding_model = PretrainedEmbeddings(
            str(path), DIMS, skip_first_line=SKIP_FIRST
        )
        return embedding_model
    raise NotImplementedError(f"Only fasttext is supported, not {PROVIDER}")


def print_embedding_summary(trained_emb_model):
    print("Shape of embedding matrix:", trained_emb_model.vectors.shape)
    print("Size of vocabulary:", len(trained_emb_model.vocab))
    print(f"UNK token {trained_emb_model.vocab.default_index}:{trained_emb_model.vocab.default_word}") 
    print(f"PAD token {trained_emb_model.vocab.index(trained_emb_model.pad_token)}:{trained_emb_model.pad_token}")
    print("First few entries in the vocabulary:")
    for _, (i, w) in zip(range(10), trained_emb_model.vocab.items()):
        print(f"{i}:{w}")
    for word in ['person', 'human', 'city', 'place', 'camera-ready']:
        print(word, word in trained_emb_model, trained_emb_model.index(word), trained_emb_model.embed_word(word).shape)


def fasttext_tokenizer(sentence: str):
    return sentence.split()


def main() -> None:
    snli, fold_ids = get_snli_dataset()
    print_snli_summary(snli, fold_ids)

    embedding_model = get_embedding_model()
    print_embedding_summary(embedding_model)

if __name__ == "__main__":
    main()
