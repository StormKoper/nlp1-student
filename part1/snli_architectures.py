import pickle
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from snli_config import DATA_PATH, TOKENIZER_PATH, ARCHITECTURES_PATH
from snli_representations import token_indicator, load_representation, load_tfidf_representation
from snli_config import K
from collections import OrderedDict
import os

# Works with all representations. 
def logistic_regression(representation_name, tfidf=False, max_iter = 100, n_jobs = -1):
    
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


def save_architecture(model, model_name) -> None:
    if not os.path.exists(ARCHITECTURES_PATH):
        os.mkdir(ARCHITECTURES_PATH)
    with open(f"{ARCHITECTURES_PATH/model_name}.pickle", 'wb') as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:

    logistic_model_token_indicator = logistic_regression("token_indicator")
    save_architecture(logistic_model_token_indicator, 'logistic_regression_token_indicator')

    logistic_model_bag_of_tokens = logistic_regression("bag_of_tokens")
    save_architecture(logistic_model_bag_of_tokens, 'logistic_regression_bag_of_tokens')

    logistic_model_token_indicator_tfidf = logistic_regression("tfidf_token_indicator", tfidf = True)
    save_architecture(logistic_model_token_indicator_tfidf, 'logistic_regression_token_indicator_tfidf')

    logistic_model_bag_of_tokens_tfidf = logistic_regression("tfidf_bag_of_tokens", tfidf = True)
    save_architecture(logistic_model_bag_of_tokens_tfidf, 'logistic_regression_bag_of_tokens_tfidf')

if __name__ == '__main__':
    main()
