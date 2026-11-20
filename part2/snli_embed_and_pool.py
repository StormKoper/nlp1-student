import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence 
from .snli_processing import get_embedding_model
import numpy as np
from datasets import load_from_disk
from .snli_config import TOKENIZATION_PATH, EMBEDDING_PATH, EMBEDDING, DIMS, SKIP_FIRST, EMB_AND_POOL_PATH
from .clslib import PretrainedEmbeddings
import pickle
import os

# reusing the tutorial's EmbedAndPoolEncoder
class EmbedAndPoolEncoder(nn.Module):
    """
    This encoder embeds a sequence of token ids and then pools the result along sequence length using
    sum or mean.
    """
    def __init__(self, embedding_matrix, pool_type: str, padding_idx=0, freeze_embeddings=True):
        """
        embedding_matrix: a torch tensor of shape (vocab_size, emb_dim)
        average_pool: use True for mean and False for sum
        padding_idx: the id of the pad token (this is for batched computation)
            suppose we are batching three sequences [1, 2, 3], [6, 2] and [2, 1, 4, 5]
            as they differ in length, we use the following trick to batch them into a single tensor
            [
                [1, 2, 3, 0],
                [6, 2, 0, 0],
                [2, 1, 4, 5]
            ]
            that is we extend the shorter sequences with 0s until they match the longest sequence
        freeze_embeddings: tells torch that we are not updating the parameters in the embedding matrix
        """
        super().__init__()
        self.pool_type = pool_type
        # Pretrained embeddingsemb_and_pool_models
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )

    @property
    def enc_dim(self):
        return 2 * self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def pool_sentence(self, ids):
        # ids: [batch, word_position]
        valid = ids.ne(self.pad_idx).unsqueeze(-1)
        vectors = self.embedding(ids)

        if self.pool_type == "max":
            pooled = vectors.masked_fill(~valid, float("-inf")).amax(dim=1)
            return pooled.masked_fill(~valid.any(dim=1), 0.0)

        summed = vectors.masked_fill(~valid, 0.0).sum(dim=1)
        if self.pool_type == "sum":
            return summed
        if self.pool_type == "average":
            word_count = valid.sum(dim=1).clamp_min(1)
            return summed / word_count

        raise ValueError(f"Unknown pooling type: {self.pool_type}")

    def forward(self, x):     
        """
        Encodes each sequence in the batch as a single vector of size emb_dim.
        
        x: token ids with shape (batch_size, max_seq_length, emb_dim)        
        """
        # x: [batch, word_position, sentence_role]
        premise = self.pool_sentence(x[..., 0])
        hypothesis = self.pool_sentence(x[..., 1])
        return torch.cat((premise, hypothesis), dim=-1)


# helper function from tutorial
def collate_sequences(sequences, padding_value=0):    
    """
    This is a helper to batch tokenized sequences of variable length into a torch tensor with shape (num_sequences, longest_seq_length)
    sequences: list of tokenized texts (each text is a list of token ids)
    padding_value: the id of the pad symbol
    """
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
    x = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=padding_value)
    return x


def save_emb_and_pool(model, size, name: str) -> None:

    if not os.path.exists(EMB_AND_POOL_PATH):
        os.mkdir(EMB_AND_POOL_PATH)

    path = f"{EMB_AND_POOL_PATH}/{name}_{str(size)}"
    with open(f"{path}.pickle", "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(name: str):
    with open(name, "rb") as file:
        model = pickle.load(file)
    return model


def main() -> None:
    # tokenizer = load_from_disk(TOKENIZATION_PATH)
    # test = torch.tensor(collate_sequences(tokenizer['train']['input_ids']))

    # embedding_model = get_embedding_model()
    path = EMBEDDING_PATH / EMBEDDING
    for size in [DIMS]:
        print(f"size: {size}")
        embedding_model = PretrainedEmbeddings(
                str(path), int(size), skip_first_line=SKIP_FIRST
        )
        for pooling_type in ['average', 'max', 'sum']:

            emb_model = EmbedAndPoolEncoder(
            torch.tensor(embedding_model.vectors, dtype=torch.float32),
            pool_type=pooling_type,
            freeze_embeddings=True, 
            padding_idx=embedding_model.vocab.index(embedding_model.pad_token)
            )
            save_emb_and_pool(emb_model, size, pooling_type)


if __name__ == '__main__':
    main()
