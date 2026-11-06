import pickle
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from snli_config import DATA_PATH, TOKENIZER_PATH, ARCHITECTURES_PATH
from snli_representations import token_indicator, load_representation, load_tfidf_representation
from snli_config import K
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import smlp
import os

# Works with all representations. 
def logistic_regression(representation_name, tfidf=False, max_iter = 5, n_jobs = -1):
    
    models = OrderedDict()

    if tfidf:
        X, y, folds = load_tfidf_representation(representation_name)

        for k in range(K):
            train_mask = folds != k
            X_train_sparse = X[k][train_mask]
            training_labels = y[train_mask]

            logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs) #sparse matrix frinedly solver 
            logistic_model.fit(X_train_sparse, training_labels)
            models[k] = logistic_model
    else:
        X, y, folds = load_representation(representation_name)

        for k in range(K):
            train_mask = folds != k
            X_train_sparse = X[train_mask]
            training_labels = y[train_mask]

            logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs) #sparse matrix frinedly solver 
            logistic_model.fit(X_train_sparse, training_labels)
            models[k] = logistic_model

    return models

def single_layer_mlp(
    representation_name,
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

    for k in range(K):
        
        train_mask = folds != k
        fold_X = X[k] if tfidf else X

        X_train_sparse = fold_X[train_mask]
        training_labels = y[train_mask]

        # Reusing the tutorial's sparse dataset and batching helpers
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
        model = smlp.SparseMLP(
            input_dim=X_train_sparse.shape[1],
            hidden_dim=hidden_dim,
            num_classes=3,
            dropout=dropout,
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
                f"Train loss: {metrics['loss']:.4f} | "
                f"Train accuracy: {metrics['accuracy']:.4f}"
            )

        # Disable dropout for later prediction; no data is evaluated here.
        model.eval()
        models[k] = model

def save_architecture(model, model_name) -> None:
    if not os.path.exists(ARCHITECTURES_PATH):
        os.mkdir(ARCHITECTURES_PATH)
    with open(f"{ARCHITECTURES_PATH/model_name}.pickle", 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:

    snli_logistic_model_token_indicator = logistic_regression("snli_token_indicator")
    save_architecture(snli_logistic_model_token_indicator, 'snli_logistic_regression_token_indicator')

    snli_logistic_model_bag_of_tokens = logistic_regression("snli_bag_of_tokens")
    save_architecture(snli_logistic_model_bag_of_tokens, 'snli_logistic_regression_bag_of_tokens')

    snli_logistic_model_token_indicator_tfidf = logistic_regression("snli_tfidf_token_indicator", tfidf = True)
    save_architecture(snli_logistic_model_token_indicator_tfidf, 'snli_tfidf_logistic_regression_token_indicator')

    snli_logistic_model_bag_of_tokens_tfidf = logistic_regression("snli_tfidf_bag_of_tokens", tfidf = True)
    save_architecture(snli_logistic_model_bag_of_tokens_tfidf, 'snli_tfidf_logistic_regression_bag_of_tokens')

    snli_mlp1_token_indicator = single_layer_mlp("snli_token_indicator")
    save_architecture(snli_mlp1_token_indicator, 'snli_mlp1_token_indicator')

    snli_mlp1_bag_of_tokens = single_layer_mlp("snli_bag_of_tokens")
    save_architecture(snli_mlp1_bag_of_tokens, 'snli_mlp1_bag_of_tokens')

    snli_mlp1_token_indicator_tfidf = single_layer_mlp("snli_tfidf_token_indicator", tfidf = True)
    save_architecture(snli_mlp1_token_indicator_tfidf, 'snli_tfidf_mlp1_token_indicator')

    snli_mlp1_bag_of_tokens_tfidf = single_layer_mlp("snli_tfidf_bag_of_tokens", tfidf = True)
    save_architecture(snli_mlp1_bag_of_tokens_tfidf, 'snli_tfidf_mlp1_bag_of_tokens')

if __name__ == '__main__':
    main()
