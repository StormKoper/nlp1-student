## Overview


| Component         | Our Choice                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| Dataset           | SNLI                                                                                                           |
| Tokenization      | BPE                                                                                                            |
| Representation 1  | Token Indicators                                                                                               |
| Representation 2  | BoW                                                                                                            |
| Representation 3  |                                                                                                                |
| Architecture 1    | Logistic Regression                                                                                            |
| Architecture 2    | Single Layer MLP                                                                                               |
| Architecture 3    |                                                                                                                |
| Validation        | 5-Fold CV                                                                                                      |
| Evaluation        | Accuracy, Precision, Recall                                                                                    |
| Bayesian Analysis | Bayesian Estimation with ROPE on Accuracy (Maybe on Precision and Recall too if we have time to generalize it) |


## Pre-Processing

Our pre-processing went as follows:
1. We cleaned up the dataset to remove examples with invalid labels or empty inputs
2. We set up a stratified $K$-fold split by adding to each training example $(x,\ell)$, the fold $k \in [0, K-1]$ for which $(x,\ell)$ is in the validation set of fold $k$. 
3. We cached the dataset locally, so that it may be used for later parts of the assignment.

We chose stratified $K$-folding over uniform $K$-folding because it is generally preferable since it reduces variation in class proportions between folds. This makes evaluating performance among the different validation sets more stable. For our first analysis we will use $K = 5$. 

We chose to set up our split purely from the training set, and for now we will not use SNLI's built in validation set in our experiments, using instead our $K$ validation sets folded from the training set.

## Tokenization

We chose to use the tutorial's tokenizer, as BPE is a reasonable choice for SNLI because it can represent uncommon or unseen words by splitting them into smaller, familiar tokens. This avoids needing a huge vocabulary of complete words while keeping sequences shorter than character-level tokenization.

We set `vocab_size=10000` to balance the number of tokens against the size of our input vectors, and `min_frequency=5` to avoid learning very rare token combinations. This is preliminary and could be adjusted. We trained BPE once on the entire training set so that all five folds use the same vocabulary. This causes some minor data leakage, since BPE sees the text from the validation folds, but it doesn't use their labels. We considered this an acceptable trade-off because we want to keep tokenization fixed and focus our evaluation on the different representations and classifiers, rather than introducing variation from separately trained tokenizers.