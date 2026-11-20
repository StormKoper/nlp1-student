import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

from . import clslib, experiment
from .snli_config import DIMS, EMBEDDING, EMBEDDING_PATH, K, ROOT, SEED, SKIP_FIRST, TOKENIZATION_PATH


# to extract the ids from the interface
class CachedTokenIDs(clslib.Tokenizer):
    """Return saved IDs through the tutorial's interface; no tokenization."""

    def ids(self, token_ids):
        return token_ids


def cross_validate(data_splits: dict, embedding_model: clslib.PretrainedEmbeddings,
          freeze_embeddings: bool, num_epochs, enc_type: str,
          labels: np.array, folds: np.array, device=None, save_to=None
        ):

    enc_kwargs = {"freeze_embeddings": freeze_embeddings}
    if enc_type == "gru":
        enc_kwargs["hidden_size"] = 128
    elif enc_type == "embed-and-pool":
        enc_kwargs["pool_type"] = "average"

    results = experiment.cross_validation(
        outcome_space=["entailment", "neutral", "contradiction"],
        data_splits=data_splits,
        tokenizer=CachedTokenIDs(),
        emb_model=embedding_model,
        enc_type=enc_type,
        enc_kwargs=enc_kwargs,
        regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
        opt_kwargs={
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 64,
            "num_epochs": num_epochs,
        },
        device=device,
        save_to=save_to / "architectures" / f"snli_{enc_type}",
    )

    # put each folds predictions back into the original snli row order.
    predictions = np.empty(len(labels), dtype=np.int64)
    for fold, fold_predictions in results["predictions"].items():
        predictions[folds == fold] = fold_predictions

    # save to cache
    prediction_path_model = save_to / "predictions" / enc_type
    prediction_path_model.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path_model / f"snli_{enc_type}.npy", predictions)
    np.save(prediction_path_model / f"snli_{enc_type}_labels.npy", labels)
    np.save(prediction_path_model / f"snli_{enc_type}_folds.npy", folds)
    accuracy = float(np.mean(predictions == labels))
    print(f"{enc_type} cross-validation accuracy: {accuracy:.4f}")
    with (save_to / "architectures" / f"snli_{enc_type}" / "metrics.json").open("w") as file:
        json.dump({"fold_accuracies": results["accuracies"], "accuracy": accuracy,
                   "seed": SEED}, file, indent=2)
    return results


# main training function. 
def main(*, num_epochs=5, freeze_embeddings=True,
         device=None, save_to=None):
    embedding_file = EMBEDDING_PATH / EMBEDDING

    # check caches
    if not TOKENIZATION_PATH.is_dir():
        raise FileNotFoundError(
            f"Cached SNLI token IDs missing: {TOKENIZATION_PATH}. "
            "Run python -m part2.snli_tokenization first."
        )
    if not embedding_file.is_file():
        raise FileNotFoundError(
            f"Cached embeddings missing: {embedding_file}. "
            "Run python -m part2.snli_processing first."
        )

    # load from cache the tokenization and embedding
    snli = load_from_disk(TOKENIZATION_PATH)
    embedding_model = clslib.PretrainedEmbeddings(
        str(embedding_file), DIMS, skip_first_line=SKIP_FIRST
    )

    # cuda if available
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_to = Path(save_to) if save_to is not None else ROOT / "data"
    torch.manual_seed(SEED)

    # keep our existing folds, the tutorial expects (input, label) examples.
    data = snli["train"]
    labels = np.asarray(data["label"])
    folds = np.asarray(data["cv_fold"])

    # turn the data into the splits
    pairs = list(zip(data["input_ids"], labels.tolist()))
    data_splits = {
        fold: {
            "training": [pairs[i] for i in np.flatnonzero(folds != fold)],
            "validation": [pairs[i] for i in np.flatnonzero(folds == fold)],
        }
        for fold in range(K)
    }

    results_gru = cross_validate(
        data_splits=data_splits,
        embedding_model=embedding_model,
        freeze_embeddings=freeze_embeddings,
        num_epochs=num_epochs,
        enc_type="gru",
        labels=labels,
        folds=folds,
        device=device,
        save_to=save_to,
    )
    results_embpool = cross_validate(
        data_splits=data_splits,
        embedding_model=embedding_model,
        freeze_embeddings=freeze_embeddings,
        num_epochs=num_epochs,
        enc_type="embed-and-pool",
        labels=labels,
        folds=folds,
        device=device,
        save_to=save_to,
    )
    return results_gru, results_embpool


if __name__ == "__main__":
    main()
