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
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):     
        """
        Encodes each sequence in the batch as a single vector of size emb_dim.
        
        x: token ids with shape (batch_size, max_seq_length, emb_dim)        
        """
        # [..., max_seq_length, emb_dim]
        e = self.embedding(x)  
        # Normalization for pooling
        if self.pool_type == 'average':
            # Total non-padding tokens (min of 1 to avoid problems when averaging the sum of embs)
            # [..., max_seq_length, 1]  # keepdim=True keeps the last axis, this is needed for division later on
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        elif self.pool_type == 'sum':  # sum pooling
            den = 1
        elif self.pool_type == 'max':  # max pooling
            return torch.max(e, dim=-2)
        else:
            raise NotImplementedError("No such pool type have been implemented.")

        # [..., emb_dim]
        h = torch.sum(e, -2) / den  # see that we reduce-sum the sequence dimension, not the representation dimension
        return h


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
