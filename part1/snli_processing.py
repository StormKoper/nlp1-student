from pathlib import Path
import numpy as np
from datasets import load_dataset, DatasetDict
import sklearn

# Config
K = 5
SEED = 42
SAVE_PATH = Path(__file__).resolve().parent / "data" / "snli" # Where the local cache will go 

# 1. Download SNLI from Hugging Face
snli = load_dataset("stanfordnlp/snli")

# 2. Prune bad examples with invalid labels or empty inputs
def valid(example):
    return (
        example["label"] in (0, 1, 2)
        and isinstance(example["premise"], str)
        and isinstance(example["hypothesis"], str)
        and bool(example["premise"].strip())
        and bool(example["hypothesis"].strip())
    )

snli = DatasetDict({
    split: data.filter(valid)
    for split, data in snli.items()
})

# 3. Assign each training example to one of K folds.
train = snli["train"]
fold_ids = np.empty(len(train), dtype=int)

kf = sklearn.model_selection.StratifiedKFold(
    n_splits=K, 
    shuffle=True, 
    random_state=SEED
) 

# We chose the representation of assiging an element the id of the fold in which it is in the validation set. 
for fold, (_, val_indices) in enumerate(kf.split(train, train["label"])):
    fold_ids[val_indices] = fold

# We only reconfigure the training dataset to have these k-fold ids. Test will be used for testing, and the designated validation set will be unused. 
snli["train"] = train.add_column(
    "cv_fold", fold_ids.tolist()
)

# 4. Save the processed dataset locally.
if SAVE_PATH.exists():
    raise FileExistsError(
        f"{SAVE_PATH} already exists. "
        "Remove it before regenerating the dataset."
    )

snli.save_to_disk(str(SAVE_PATH)) 

# 5. Display a summary.
for split, data in snli.items():
    print(f"{split}: {len(data):,} examples")

for fold in range(K):
    count = np.sum(fold_ids == fold)
    print(f"Fold {fold}: {count:,} examples")

print(f"Saved to {SAVE_PATH}")

