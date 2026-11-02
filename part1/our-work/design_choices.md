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

We chose stratified $K$-folding over uniform $K$-folding because it is generally preferable since it reduces variation in class proportions between folds. This makes evaluating performance among the different validation sets more stable. 

##