import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import StratifiedKFold
from snli_config import DATA_PATH, SEED, K


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

def main() -> None:
    # Check whether the dataset already exists
    if DATA_PATH.exists():
        raise FileExistsError(
            f"{DATA_PATH} already exists. "
            "Remove it before regenerating the dataset."
        )

    # 1. Download SNLI from Hugging Face
    snli = load_dataset("stanfordnlp/snli")

    # 2. Prune bad examples with invalid labels or empty inputs
    snli = DatasetDict({
        split: data.filter(valid)
        for split, data in snli.items()
    })

    # 3. Assign each training example to one of K folds.
    snli["train"], fold_ids = assign_folds(snli["train"])

    snli["test"] = concatenate_datasets([snli["validation"], snli["test"]])
    del snli["validation"]

    # 4. Save the processed dataset locally
    snli.save_to_disk(str(DATA_PATH)) 

    # 5. Print a summary
    for split, data in snli.items():
        print(f"{split}: {len(data):,} examples")

    for fold in range(K):
        count = np.sum(fold_ids == fold)
        print(f"Fold {fold}: {count:,} examples")

    print(f"Saved to {DATA_PATH}")

if __name__ == "__main__":
    main()
