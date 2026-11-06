import pickle
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from snli_config import DATA_PATH, TOKENIZER_PATH, ARCHITECTURES_PATH
from snli_representations import token_indicator
from snli_config import K
from collections import OrderedDict
import os


def logistic_regression(dataset, tokenizer, max_iter = 100, n_jobs = -1):
    models = OrderedDict()

    for k in range(K):
        train = dataset['train']
        training_set = train.filter(lambda x: x["cv_fold"] == k)

        X_train_sparse, training_labels, _ = token_indicator(training_set, tokenizer)
        training_labels = training_labels.tolist()    

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
    # Load our saved dataset
    snli = load_from_disk(DATA_PATH)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH/ "bpe.json"))

    logistic_model = logistic_regression(snli, tokenizer)
    save_architecture(logistic_model, 'logistic_regression')


if __name__ == '__main__':
    main()
