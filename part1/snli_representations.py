import numpy as np
from datasets import load_from_disk
from scipy.sparse import csr_matrix, load_npz, save_npz
from snli_config import DATA_PATH, REPRESENTATIONS_PATH, TOKENIZER_PATH, TOKENIZERS
from tokenizers import Tokenizer
from tqdm.auto import tqdm


# Token indicator representation
def token_indicator(data, tokenizer):
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

    for i, example in enumerate(tqdm(data, desc="Encoding SNLI - token indicator")):
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
        shape=(len(data), 2 * V)
    )

    return X

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

    for i, example in enumerate(tqdm(train_set, desc="Encoding SNLI - bag of tokens")):
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

    return X

# Convert token indicator or bag of tokens to term frequency inverse document frequency
def convert_to_tfidf(X, reference):
    """
    Convert count (indicator or bag of words) term frequency to
    term frequency inverse document frequency.
    The documents are the premise-hypothesis pairs in the dataset.
    Uses smoothed TF-IDF computation (+1 document and document frequency).
    """
    df = reference.getnnz(axis=0)
    idf = np.log((reference.shape[0] + 1) / (df + 1)) + 1
    tfidf = X.multiply(idf).tocsr()
    return tfidf

# Use the conversion for the folds
def convert_to_tfidf_folds(X, folds):
    """
    Convert count (indicator or bag of words) term frequency to
    term frequency inverse document frequency.
    Uses the folds argument to define the reference data for the document frequency.
    """
    tfidf_folds = []
    for fold in np.unique(folds):
        train = X[folds != fold]
        tfidf = convert_to_tfidf(X, train)
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
    save_path = REPRESENTATIONS_PATH / path_name

    # Save it
    save_path.mkdir(parents=True, exist_ok=True)
    if isinstance(X, csr_matrix):
        save_npz(save_path / "X.npz", X)
    elif isinstance(X, list):
        for fold in np.unique(folds):
            save_npz(save_path / f"X_fold_{fold}.npz", X[fold])
    else:
        raise TypeError("X must be a csr_matrix or list of csr_matrices")
    np.save(save_path / "y.npy", y)
    if folds is not None:
        np.save(save_path / "folds.npy", folds)

    # Logging Output
    features = X.shape[1] if isinstance(X, csr_matrix) else X[0].shape[1]
    print(f"Saved representation to {save_path}: {features:,} features")

# Helper function to load the representation for faster testing
def load_representation(path_name):
    load_path = REPRESENTATIONS_PATH / path_name
    X = load_npz(load_path / "X.npz")
    y = np.load(load_path / "y.npy")
    folds = np.load(load_path / "folds.npy")
    return X, y, folds

def load_tfidf_representation(path_name): 
    load_path = REPRESENTATIONS_PATH / path_name
    folds = np.load(load_path / "folds.npy")
    X = {
        fold: load_npz(load_path / f"X_fold_{fold}.npz")
        for fold in sorted(np.unique(folds))
    }
    y = np.load(load_path / "y.npy")
    return X, y, folds


def main() -> None:
    # Load our saved dataset
    snli = load_from_disk(DATA_PATH)
    train = snli["train"] # All training data regardless of fold
    y_train = np.asarray(train["label"])
    folds = np.asarray(train["cv_fold"])
    print("Computing representations for the training set")
    for tokenizer_name in TOKENIZERS:
        print(f"\n--- Using tokenizer {tokenizer_name} ---\n")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / f"{tokenizer_name}_bpe.json"))

        # Compute and save the representations
        X_train_ti = token_indicator(train, tokenizer)
        save_representation(X_train_ti, y_train, folds, f"{tokenizer_name}_token_indicator")

        X_train_bot = bag_of_tokens(train, tokenizer)
        save_representation(X_train_bot, y_train, folds, f"{tokenizer_name}_bag_of_tokens")

        X_train_tfidf_ti_folds = convert_to_tfidf_folds(X_train_ti, folds)
        save_representation(X_train_tfidf_ti_folds, y_train, folds, f"{tokenizer_name}_tfidf_token_indicator")

        X_train_tfidf_bot_folds = convert_to_tfidf_folds(X_train_bot, folds)
        save_representation(X_train_tfidf_bot_folds, y_train, folds, f"{tokenizer_name}_tfidf_bag_of_tokens")

    test = snli["test"]
    y_test = np.asarray(test["label"])
    print("\nComputing representations for the testing set")
    for tokenizer_name in TOKENIZERS:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH / f"{tokenizer_name}_bpe.json"))

        X_test_ti = token_indicator(test, tokenizer)
        save_representation(X_test_ti, y_test, None, f"{tokenizer_name}_token_indicator_test")

        X_test_bot = bag_of_tokens(test, tokenizer)
        save_representation(X_test_bot, y_test, None, f"{tokenizer_name}_bag_of_tokens_test")

        X_test_tfidf_ti = convert_to_tfidf(X_test_ti, X_train_ti)
        save_representation(X_test_tfidf_ti, y_test, None, f"{tokenizer_name}_tfidf_token_indicator_test")

        X_test_tfidf_bot = convert_to_tfidf(X_test_bot, X_train_bot)
        save_representation(X_test_tfidf_bot, y_test, None, f"{tokenizer_name}_tfidf_bag_of_tokens_test")


if __name__ == "__main__":
    main()
