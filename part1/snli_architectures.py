import numpy as np
from datasets import load_from_disk, load_dataset
from scipy.sparse import csr_matrix, save_npz, load_npz
from tokenizers import Tokenizer
from tqdm.auto import tqdm
import sys
from sklearn.linear_model import LogisticRegression

from snli_config import DATA_PATH, TOKENIZER_PATH, REPRESENTATIONS_PATH
from snli_representations import load_representation, token_indicator
from experiment import *
from loglin import *
from snli_config import K


def logistic_regression(dataset, tokenizer, max_iter = 100, n_jobs = -1):
    models, predictions, accuracies, precitions = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    for k in range(K):
        train = dataset['train']
        training_set = train.filter(lambda x: x["cv_fold"] == k)

        X_train_sparse, training_labels, _ = token_indicator(training_set, tokenizer)
        training_labels = training_labels.tolist()    

        logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs) #sparse matrix frinedly solver 
        logistic_model.fit(X_train_sparse, training_labels)
        models[k] = logistic_model

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}


def main() -> None:
    # Load our saved dataset
    snli = load_from_disk(DATA_PATH)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    logit_results = logistic_regression(snli, tokenizer)
    print(logit_results['model'][0])
    print(logit_results['predictions'][0])
    print(logit_results['accuracies'][0])


if __name__ == '__main__':
    main()

