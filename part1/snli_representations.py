import numpy as np
from datasets import load_from_disk
from scipy.sparse import csr_matrix, load_npz, save_npz, vstack
from snli_config import DATA_PATH, REPRESENTATIONS_PATH, TOKENIZER_PATH, K
from tokenizers import Tokenizer
from tqdm.auto import tqdm


# Token indicator representation
def token_indicator(train_set, tokenizer):
    """
    Encode all SNLI premise-hypothesis pairs into a sparse feature matrix
    with token indicator encoding.

    Each example is represented by two feature vectors concatenated together, one for the
    premise and one for the hypothesis.

    Args:
        train_set: The SNLI dataset containing premises, hypotheses,
            labels, and cross-validation fold assignments.

    Returns:
        X: Sparse feature matrix with one row per example.
        y: Array of classification labels.
        folds: Array of cross-validation fold assignments.
    """

    V = tokenizer.get_vocab_size()
    rows = []
    cols = []

    for i, example in enumerate(tqdm(train_set, desc="Encoding SNLI - Indicator")):
        # Remove duplicate token IDs so each token is counted only once, using set. This makes it binary and not bag of words. 
        premise_ids = set(tokenizer.encode(example["premise"]).ids)
        hypothesis_ids = set(tokenizer.encode(example["hypothesis"]).ids)

        for token_id in premise_ids:
            rows.append(i) # The row is the id
            cols.append(token_id) # The column list tells which tokens are included in the examples

        # Hypothesis features: columns V to 2V-1
        for token_id in hypothesis_ids:
            rows.append(i) # The row is the id
            cols.append(token_id + V) # Columns tell which tokens are included again, but this time we add V because it is the second feature set, since we distinguish counts for premises and hypotheses in our representation.

    # Directly construct a sparse matrix
    X = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(train_set), 2 * V)
    )

    # All three of these are indexed in the same order by construction
    y = np.asarray(train_set["label"])
    folds = np.asarray(train_set["cv_fold"])

    return X, y, folds

# Token bag of words representation
def bag_of_tokens(train_set, tokenizer):
    """
    Encode all SNLI premise-hypothesis pairs into a sparse feature matrix
    with token BoW encoding.

    Each example is represented by two feature vectors concatenated
    together, one for the premise and one for the hypothesis.

    Args:
        train_set: The SNLI dataset containing premises, hypotheses,
            labels, and cross-validation fold assignments.

    Returns:
        X: Sparse feature matrix with one row per example.
        y: Array of classification labels.
        folds: Array of cross-validation fold assignments.
    """

    V = tokenizer.get_vocab_size()
    rows = []
    cols = []

    for i, example in enumerate(tqdm(train_set, desc="Encoding SNLI  - Bag of Words")):
        premise_ids = tokenizer.encode(example["premise"]).ids
        hypothesis_ids = tokenizer.encode(example["hypothesis"]).ids

        for token_id in premise_ids:
            rows.append(i) # The row is the id
            cols.append(token_id) # The column list tells which tokens are included in the examples

        # Hypothesis features: columns V to 2V-1
        for token_id in hypothesis_ids:
            rows.append(i) # The row is the id
            cols.append(token_id + V) # Columns tell which tokens are included again, but this time we add V because it is the second feature set, since we distinguish counts for premises and hypotheses in our representation.

    # Directly construct a sparse matrix
    X = csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(train_set), 2 * V)
    )

    # All three of these are indexed in the same order by construction
    y = np.asarray(train_set["label"])
    folds = np.asarray(train_set["cv_fold"])

    return X, y, folds

# Convert token indicator or bag of tokens to term frequency inverse document frequency
def convert_to_tfidf(X, folds):
    """
    Convert count (indicator or bag of words) term frequency to
    term frequency inverse document frequency.
    The documents are the premise-hypothesis pairs in the dataset.
    Uses smoothed TF-IDF computation (+1 document and document frequency).
    """
    tfidf_folds = []
    for fold in np.unique(folds):
        train = X[folds != fold]
        df = train.getnnz(axis=0)
        idf = np.log((train.shape[0] + 1) / (df + 1)) + 1
        tfidf = X.multiply(idf).tocsr()
        tfidf_folds.append(tfidf)
    return tfidf_folds

# Helper function to save the representation for modelling
def save_representation(X, y, folds, path_name):
    """
    Save a computed representation and its associated labels and folds.

    Creates a subdirectory under REPRESENTATIONS_PATH and stores the sparse
    feature matrix and corresponding NumPy arrays for later reuse.

    Args:
        X: Sparse feature matrix.
        y: Array of classification labels.
        folds: Array of cross-validation fold assignments.
        path_name: Name of the subdirectory in which to save the files.
    """
    LOCAL_PATH = REPRESENTATIONS_PATH / path_name

    # Save it
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    if isinstance(X, csr_matrix):
        save_npz(LOCAL_PATH / "X.npz", X)
    elif isinstance(X, list):
        for fold in np.unique(folds):
            save_npz(LOCAL_PATH / f"X_fold_{fold}.npz", X[fold])
    else:
        raise TypeError("X must be a csr_matrix or list of csr_matrices")
    np.save(LOCAL_PATH / "y.npy", y)
    np.save(LOCAL_PATH / "folds.npy", folds)

    # Logging Output
    if isinstance(X, list):
        X = X[0]
    print(f"Saved {X.shape[0]:,} examples, {X.shape[1]:,} features to {REPRESENTATIONS_PATH}")
    for fold in sorted(np.unique(folds)):
        print(f"Fold {fold}: {(folds != fold).sum():,} training examples")

# Helper function to load the representation for faster testing
def load_representation(path_name):
    LOCAL_PATH = REPRESENTATIONS_PATH / path_name
    X = load_npz(LOCAL_PATH / "X.npz")
    y = np.load(LOCAL_PATH / "y.npy")
    folds = np.load(LOCAL_PATH / "folds.npy")
    return X, y, folds

def load_tfidf_representation(path_name): 
    LOCAL_PATH = REPRESENTATIONS_PATH / path_name
    folds = np.load(LOCAL_PATH / "folds.npy")
    X = {
        fold: load_npz(LOCAL_PATH / f"X_fold_{fold}.npz")
        for fold in sorted(np.unique(folds))
    }
    y = np.load(LOCAL_PATH / "y.npy")
    return X, y, folds


def main() -> None:
    # Load our saved dataset
    snli = load_from_disk(DATA_PATH)
    train_full = snli["train"] # All training data regardless of fold
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / "bpe.json"))

    # Compute and save the representations
    X_ti, y_ti, folds_ti = token_indicator(train_full, tokenizer)
    save_representation(X_ti, y_ti, folds_ti, "token_indicator")

    X_bot, y_bot, folds_bot = bag_of_tokens(train_full, tokenizer)
    save_representation(X_bot, y_bot, folds_bot, "bag_of_tokens")

    tfidf_folds_ti = convert_to_tfidf(X_ti, folds_ti)
    save_representation(tfidf_folds_ti, y_ti, folds_ti, "tfidf_token_indicator")

    tfidf_folds_bot = convert_to_tfidf(X_bot, folds_bot)
    save_representation(tfidf_folds_bot, y_bot, folds_bot, "tfidf_bag_of_tokens")


if __name__ == "__main__":
    main()
