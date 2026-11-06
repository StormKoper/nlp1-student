import os
import pickle
from collections import OrderedDict

import numpy as np
import smlp
import torch
from sklearn.linear_model import LogisticRegression
from snli_config import ARCHITECTURES_PATH, PREDICTIONS_PATH, K
from snli_representations import load_representation, load_tfidf_representation
from torch import nn
from torch.utils.data import DataLoader


class n_layer_SparseMLP(nn.Module):
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2, layers=1):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        args = ()
        if layers > 1:
            for _ in range(1, layers):
                args += (nn.Linear(hidden_dim, hidden_dim), nn.ReLU())


        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            *args,
            nn.Linear(hidden_dim, num_classes),            
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )


    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)


# Works with all representations. 
def logistic_regression(representation_name, tfidf=False, max_iter = 500):
    
    models = OrderedDict()

    if tfidf:
        X, y, folds = load_tfidf_representation(representation_name)
    else:
        X, y, folds = load_representation(representation_name)
    preds = np.zeros_like(y)

    for k in range(K):
        train_mask = folds != k
        fold_X = X[k] if tfidf else X
        X_train_sparse = fold_X[train_mask]
        training_labels = y[train_mask]

        logistic_model = LogisticRegression(solver="saga", max_iter=max_iter) #sparse matrix frinedly solver 
        logistic_model.fit(X_train_sparse, training_labels)
        pred = logistic_model.predict(fold_X[~train_mask])
        models[k] = logistic_model
        preds[~train_mask] = pred
        fold_acc = np.mean(preds[~train_mask] == y[~train_mask])
        print(f"Fold {k}: accuracy {fold_acc:.4f} ({representation_name})")

    acc = np.mean(preds == y)
    print(f"Logistic regression accuracy {acc:.4f} ({representation_name})")

    return models, preds

def n_layer_mlp(
    representation_name,
    layers=1,
    tfidf=False,
    hidden_dim=128,
    dropout=0.1,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_epochs=2,
    device="cpu",
):
    device = torch.device(device)
    models = OrderedDict()

    # Load cached representations
    if tfidf:
        X, y, folds = load_tfidf_representation(representation_name)
    else:
        X, y, folds = load_representation(representation_name)
    preds = np.zeros_like(y)

    for k in range(K):
        
        train_mask = folds != k
        fold_X = X[k] if tfidf else X

        X_train_sparse = fold_X[train_mask]
        training_labels = y[train_mask]

        # Reusing the tutorial"s sparse dataset and batching helpers
        train_dataset = smlp.SparseDataset(
            X_train_sparse, training_labels
        )

        # Splits the dataset into batches for training via a DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=smlp.sparse_collate,
        )

        # Start each fold with a fresh model and optimizer.
        model = n_layer_SparseMLP(
            input_dim=X_train_sparse.shape[1],
            hidden_dim=hidden_dim,
            num_classes=3,
            dropout=dropout,
            layers=layers
        ).to(device)

        loss = torch.nn.NLLLoss() # We use negative log-likelihood loss

        # optimize with AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        for epoch in range(num_epochs):
            metrics = smlp.train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss,
                device
            )

            print(
                f"Fold {k + 1}/{K} | "
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train loss: {metrics["loss"]:.4f} | "
                f"Train accuracy: {metrics["accuracy"]:.4f}"
            )

        # Disable dropout for later prediction; no data is evaluated here.
        model.eval()

        test_dataset = smlp.SparseDataset(
            fold_X[~train_mask], y[~train_mask]
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=smlp.sparse_collate,
        )

        prediction_batches = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                log_probs = model(X_batch.to(device))
                predictions = log_probs.argmax(dim=1)
                prediction_batches.append(predictions.cpu().numpy())
        predictions = np.concatenate(prediction_batches)
        preds[~train_mask] = predictions
        models[k] = model
        fold_acc = np.mean(preds[~train_mask] == y[~train_mask])
        print(f"{layers}-layer MLP fold {k}: accuracy {fold_acc:.4f} ({representation_name})")

    acc = np.mean(preds == y)
    print(f"{layers}-layer MLP accuracy {acc:.4f} ({representation_name})")

    return models, preds

def save_architecture(model, model_name) -> None:
    if not os.path.exists(ARCHITECTURES_PATH):
        os.mkdir(ARCHITECTURES_PATH)
    with open(f"{ARCHITECTURES_PATH/model_name}.pickle", "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_predictions(preds, representation_name) -> None:
    if not os.path.exists(PREDICTIONS_PATH):
        os.mkdir(PREDICTIONS_PATH)
    np.save(PREDICTIONS_PATH / (representation_name + ".npy"), preds)


def main() -> None:
    print("Logistic regression")
    snli_logistic_model_token_indicator, preds = logistic_regression("snli_token_indicator")
    save_architecture(snli_logistic_model_token_indicator, "snli_logistic_regression_token_indicator")
    save_predictions(preds, "snli_logistic_regression_token_indicator")

    snli_logistic_model_bag_of_tokens_tfidf, preds = logistic_regression("snli_tfidf_bag_of_tokens", tfidf = True)
    save_architecture(snli_logistic_model_bag_of_tokens_tfidf, "snli_tfidf_logistic_regression_bag_of_tokens")
    save_predictions(preds, "snli_tfidf_logistic_regression_bag_of_tokens")

    mistral_logistic_model_bag_of_tokens_tfidf, preds = logistic_regression("mistral_tfidf_bag_of_tokens", tfidf = True)
    save_architecture(mistral_logistic_model_bag_of_tokens_tfidf, "mistral_tfidf_logistic_regression_bag_of_tokens")
    save_predictions(preds, "mistral_tfidf_logistic_regression_bag_of_tokens")
    
    # print("Single layer MLP")
    # snli_mlp1_bag_of_tokens_tfidf, preds = n_layer_mlp("snli_tfidf_bag_of_tokens", tfidf = True)
    # save_architecture(snli_mlp1_bag_of_tokens_tfidf, "snli_tfidf_mlp1_bag_of_tokens")
    # save_predictions(preds, "snli_tfidf_mlp1_bag_of_tokens")

    # snli_mlp1_bag_of_tokens_tfidf, preds = n_layer_mlp("mistral_tfidf_bag_of_tokens", tfidf = True)
    # save_architecture(snli_mlp1_bag_of_tokens_tfidf, "mistral_tfidf_mlp1_bag_of_tokens")
    # save_predictions(preds, "mistral_tfidf_mlp1_bag_of_tokens")

    print("Double layer MLP")
    snli_mlp2_bag_of_tokens_tfidf, preds = n_layer_mlp("snli_tfidf_bag_of_tokens", tfidf = True, layers=2)
    save_architecture(snli_mlp2_bag_of_tokens_tfidf, "snli_tfidf_mlp2_bag_of_tokens")
    save_predictions(preds, "snli_tfidf_mlp2_bag_of_tokens")

    mistral_mlp2_bag_of_tokens_tfidf, preds = n_layer_mlp("mistral_tfidf_bag_of_tokens", tfidf = True, layers=2)
    save_architecture(mistral_mlp2_bag_of_tokens_tfidf, "mistral_tfidf_mlp2_bag_of_tokens")
    save_predictions(preds, "mistral_tfidf_mlp2_bag_of_tokens")

if __name__ == "__main__":
    main()
