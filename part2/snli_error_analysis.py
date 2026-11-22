import numpy as np
from .snli_config import PREDICTIONS_PATH, DATA_PATH
from datasets import load_from_disk


def load_preds(path: str, model_type: str):

    if model_type == 'gru':
        model_type = model_type.upper()

    predictions = path / f"snli_{model_type}.npy"
    folds = path / f"snli_{model_type}_folds.npy"
    labels = path / f"snli_{model_type}_labels.npy"
    return np.load(predictions), np.load(labels), np.load(folds)


def sentence_length_analysis(snli_train):
    lengths = [10, 15, 20, 25]
    for lengh in lengths:
        filtered_dataset = snli_train.filter(
            lambda example: lengh-5 <= len(example["premise"].split()) <= lengh
        )
        labels = np.asarray(filtered_dataset['label'])
        pred_emb = np.asarray(filtered_dataset['predictions embed-and-pool'])
        pred_gru = np.asarray(filtered_dataset['predictions GRU'])
        correct_emb = float(np.mean(pred_emb == labels))
        correct_gru = float(np.mean(pred_gru == labels))

        print(f"Sentence length between of premise {lengh - 5} and {lengh}:")
        print(f"Accuracy gru: {correct_gru}")
        print(f"Accuracy embed and pool: {correct_emb}")
        print('\n')

        filtered_dataset = snli_train.filter(
            lambda example: lengh-5 <= len(example["hypothesis"].split()) <= lengh
        )
        labels = np.asarray(filtered_dataset['label'])
        pred_emb = np.asarray(filtered_dataset['predictions embed-and-pool'])
        pred_gru = np.asarray(filtered_dataset['predictions GRU'])
        correct_emb = float(np.mean(pred_emb == labels))
        correct_gru = float(np.mean(pred_gru == labels))

        print(f"Sentence length of hyphotheis between {lengh - 5} and {lengh}:")
        print(f"Accuracy gru: {correct_gru}")
        print(f"Accuracy embed and pool: {correct_emb}")
        print('\n')


def label_analysis(snli_train):
    for label in [0, 1, 2]:
        filtered_dataset_entailment = snli_train.filter(
            lambda example: example["label"] == label
        )
        labels = np.asarray(filtered_dataset_entailment['label'])
        pred_emb = np.asarray(filtered_dataset_entailment['predictions embed-and-pool'])
        pred_gru = np.asarray(filtered_dataset_entailment['predictions GRU'])
        correct_emb = float(np.mean(pred_emb == labels))
        correct_gru = float(np.mean(pred_gru == labels))

        if label == 0:
            class_name = 'entailment'
        elif label == 1:
            class_name = 'neutral'
        elif label == 2:
            class_name = 'contradiction'
        print(f"Accuracy for predicting {class_name} class:")
        print(f"GRU: {correct_gru}")
        print(f"Embed and pool: {correct_emb}")
        print('\n')


def main():
    print("Getting original data....")
    snli = load_from_disk(DATA_PATH)

    print("Getting predictions and comparisons for embed and pool")
    model_type = "embed-and-pool"
    emb_and_pool = PREDICTIONS_PATH / model_type
    predictions_emb, _, _= load_preds(emb_and_pool, model_type)

    snli['train'] = snli['train'].add_column("predictions embed-and-pool", predictions_emb)

    print("Getting predictions and comparisons for GRU")
    model_type = "gru"
    gru = PREDICTIONS_PATH / model_type
    predictions_gru, _, _ = load_preds(gru, model_type)

    snli['train'] = snli['train'].add_column("predictions GRU", predictions_gru)

    sentence_length_analysis(snli['train'])
    label_analysis(snli['train'])


if __name__ == '__main__':
    main()
