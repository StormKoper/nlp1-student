import pickle
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from snli_config import DATA_PATH, TOKENIZER_PATH, ARCHITECTURES_PATH
from snli_representations import token_indicator, load_representation, load_tfidf_representation
from snli_config import K
import torch
from torch.utils.data import DataLoader
import smlp
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import csv
from scipy.sparse import load_npz
from snli_config import REPRESENTATIONS_PATH

# To evaluate all of our models 

# To evaluate _._._._.test i.e. mistral_bag_of_tokens_test

# We expect the test matricies to be CSR already
def evaluate_with_metrics(
    model,
    X_test, #CSR
    y_test,
    batch_size=64,
    device="cpu",
):
    if isinstance(model, LogisticRegression):
        predictions = model.predict(X_test)

    else: 
        model = model.to(device)
        model.eval() # just for extra security

        test_loader = DataLoader(
            smlp.SparseDataset(X_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=smlp.sparse_collate, # Need the custom function for CSR
        )

        prediction_batches = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                log_probs = model(X_batch.to(device))
                predictions = log_probs.argmax(dim=1)
                prediction_batches.append(predictions.cpu().numpy())

        predictions = np.concatenate(prediction_batches)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)

    return predictions, accuracy, precision, recall

def main() -> None:

    output_dir = ARCHITECTURES_PATH.parent / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [] # making a model evaluation list 

    def evaluate_saved(model_name, test_name, tfidf=False):
        
        # load the models
        with open(ARCHITECTURES_PATH / f"{model_name}.pickle", "rb") as file:
            models = pickle.load(file)

        # Load up the test data
        test_path = REPRESENTATIONS_PATH / test_name
        y_test = np.load(test_path / "y.npy")

        if not tfidf:
            X_test = load_npz(test_path / "X.npz").tocsr()

        for fold, model in models.items():
            if tfidf:
                X_test = load_npz(
                    test_path / f"X_fold_{fold}.npz"
                ).tocsr()

            predictions, accuracy, precision, recall = evaluate_with_metrics(
                model, X_test, y_test
            )

            prediction_file = f"{model_name}_fold_{fold}_predictions.npy"
            np.save(output_dir / prediction_file, predictions)

            rows.append({
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "predictions_file": prediction_file,
            })

            print(f"{model_name}, fold {fold}: accuracy={accuracy:.4f}")

        return

    # Logistic regression
    evaluate_saved(
        "snli_logistic_regression_token_indicator",
        "snli_token_indicator_test",
    )
    evaluate_saved(
        "snli_logistic_regression_bag_of_tokens",
        "snli_bag_of_tokens_test",
    )
    evaluate_saved(
        "snli_tfidf_logistic_regression_token_indicator",
        "snli_tfidf_token_indicator_test",
        tfidf=True,
    )
    evaluate_saved(
        "snli_tfidf_logistic_regression_bag_of_tokens",
        "snli_tfidf_bag_of_tokens_test",
        tfidf=True,
    )

    # Single-hidden-layer MLP
    evaluate_saved(
        "snli_mlp1_token_indicator",
        "snli_token_indicator_test",
    )
    evaluate_saved(
        "snli_mlp1_bag_of_tokens",
        "snli_bag_of_tokens_test",
    )
    evaluate_saved(
        "snli_tfidf_mlp1_token_indicator",
        "snli_tfidf_token_indicator_test",
        tfidf=True,
    )
    evaluate_saved(
        "snli_tfidf_mlp1_bag_of_tokens",
        "snli_tfidf_bag_of_tokens_test",
        tfidf=True,
    )

    metrics_path = output_dir / "metrics.csv"

    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "fold",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "predictions_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics and prediction vectors to {output_dir}")

if __name__ == '__main__':
    main()
