from pathlib import Path

import arviz as az
import arviz_plots as azp
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from snli_config import DATA_PATH, PREDICTIONS_PATH, K


def analysis_cv_diff(cv_acc_diff, name_a, name_b):
    with pm.Model() as model:
        print("bayesian estimation")
        # Mean difference prior mu with expectation set to 0 and double the observed standard deviation
        # This makes it a weakly informative prior based on the observed data
        mu = pm.Normal("mu", 
                       mu=0, 
                       sigma=max(0.01, 2 * np.sqrt(np.mean(cv_acc_diff ** 2)))
        )
        # Standard deviation of difference must be positive, therefore a half-normal distribution is used
        # Again, the standard deviation is set to double the observed standard deviation
        # Supporting the weakly informative prior viewpoint
        sigma = pm.HalfNormal("sigma", sigma=np.std(cv_acc_diff) * 2)
        # Cross-validation with a small amount of folds is vulnerable to outliers
        # The student-t distribution is more robust to outliers due to its normality parameter nu
        nu = pm.Exponential("nu", 1 / 29.0) + 1
        likelihood = pm.StudentT("y", mu=mu, sigma=sigma, nu=nu, observed=cv_acc_diff)
        # Draw samples from a the probability distribution using Markov chain Monte Carlo sampling
        # This forms the posterior distribution by combining the prior and the likelihood
        trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)

        print("ROPE")
        # The dataset has many, relatively short examples (550K examples with 22.4 words per examples on average)
        # This includes the premise (14.1 words on average) and the hypothesis (8.3 words on average)
        # Additionally, the task is relatively straightforward, due to it resting on logical inference
        # Based on this, we expect accuracies to be relatively high
        # In turn, that results in us choosing a relatively small ROPE interval
        # This is because when accuracies are already high, a small difference is much more impactful
        # Additionally, the annotators achieved majority consensus in 98% of examples
        # The 2% in which they did not have been filtered out by our processing
        # This again shows it is absolutely possible to achieve a high accuracy and support our choice for the ROPE

        rope_interval = [-0.01, 0.01]
        azp.plot_dist(trace, var_names=["mu"])
        plt.gcf().set_size_inches(6, 3)
        posterior_mu = trace.posterior["mu"].values.flatten()
        hdi_interval = az.hdi(posterior_mu, prob=0.95)
        mean_val = np.mean(posterior_mu)
        ax = plt.gca()
        ax.axvspan(
            rope_interval[0],
            rope_interval[1],
            color="red",
            alpha=0.2,
            label=f"ROPE {rope_interval}",
        )
        ax.axvline(
            rope_interval[0], color="red", linestyle="--", alpha=0.6, linewidth=1.5
        )
        ax.axvline(
            rope_interval[1], color="red", linestyle="--", alpha=0.6, linewidth=1.5
        )
        ax.set_title(
            f"Bayesian estimation: difference in {K}-fold cross-validation accuracy\n"
            f"{name_a.replace("_", " ").capitalize()} - {name_b.replace("_", " ").capitalize()}",
            fontsize=12,
            pad=15,
        )
        ax.set_xlabel(
            "Difference in Accuracy Score (mu)", fontsize=10, labelpad=10
        )
        ax.set_ylabel("Posterior Density", fontsize=10)
        ax.legend(loc="upper right", frameon=True)
        print(f"Mean Difference: {mean_val:.5f}")
        print(f"95% HDI: [{hdi_interval[0]:.5f}, {hdi_interval[1]:.5f}]")
        print(f"ROPE Interval: {rope_interval}")

        plt.tight_layout()
        plt.savefig(f"part1/figures/BE_{name_a}-{name_b}.png", dpi=400)

def main():
    print("getting labels and folds")
    snli = load_from_disk(DATA_PATH)
    cv_labels = np.array(snli["train"]["label"])
    cv_folds = np.array(snli["train"]["cv_fold"])
    del snli

    print("getting predictions and comparisons")
    predictions = sorted(Path(PREDICTIONS_PATH).glob("*.npy"))
    predictions = {p.stem: np.load(p) for p in predictions}

    comparisons = [
        ("snli_LR_token_indicator",
         "snli_LR_tfidf_bag_of_tokens"),
        ("snli_LR_tfidf_bag_of_tokens",
         "mistral_LR_tfidf_bag_of_tokens"),
        ("snli_MLP1_tfidf_bag_of_tokens",
         "mistral_MLP1_tfidf_bag_of_tokens"),
        ("snli_MLP2_tfidf_bag_of_tokens",
         "mistral_MLP2_tfidf_bag_of_tokens"),
    ]

    tokenizers = ["snli", "mistral"]
    representation = "tfidf_bag_of_tokens"
    models = ["LR", "MLP1", "MLP2"]
    for tokenizer in tokenizers:
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_a, model_b = models[i], models[j]
                name_a = f"{tokenizer}_{model_a}_{representation}"
                name_b = f"{tokenizer}_{model_b}_{representation}"
                comparisons.append((name_a, name_b))

    print("computing metrics")
    diffs = []
    for (name_a, name_b) in comparisons:
        pred_a, pred_b = predictions[name_a], predictions[name_b]
        cv_acc_diff = np.zeros(len(np.unique(cv_folds)))
        for fold in np.unique(cv_folds):
            mask = cv_folds == fold
            fold_labels = cv_labels[mask]
            fold_pred_a = pred_a[mask]
            fold_pred_b = pred_b[mask]
            fold_acc_a = accuracy_score(fold_labels, fold_pred_a)
            fold_acc_b = accuracy_score(fold_labels, fold_pred_b)
            cv_acc_diff[fold] = fold_acc_a - fold_acc_b
        diffs.append(cv_acc_diff)

    print("running Bayesian estimation")
    for (name_a, name_b), diff in zip(comparisons, diffs):
        analysis_cv_diff(diff, name_a, name_b)

if __name__ == "__main__":
    main()
