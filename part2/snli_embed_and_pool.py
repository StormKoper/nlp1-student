import torch
import torch.nn as nn
from .snli_processing import get_embedding_model


# reusing the tutorial's EmbedAndPoolEncoder
class EmbedAndPoolEncoder(nn.Module):
    """
    This encoder embeds a sequence of token ids and then pools the result along sequence length using
    sum or mean.
    """
    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
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
        self.average_pool = average_pool
        # Pretrained embeddings
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
        if self.average_pool:
            # Total non-padding tokens (min of 1 to avoid problems when averaging the sum of embs)
            # [..., max_seq_length, 1]  # keepdim=True keeps the last axis, this is needed for division later on
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        else:  # sum pooling
            den = 1
        # [..., emb_dim]
        h = torch.sum(e, -2) / den  # see that we reduce-sum the sequence dimension, not the representation dimension
        return h


def main() -> None:
    embedding_model = get_embedding_model()

    pooled_emb_encoder = EmbedAndPoolEncoder(
    torch.tensor(embedding_model.vectors, dtype=torch.float32), 
    freeze_embeddings=True, 
    padding_idx=embedding_model.vocab.index(embedding_model.pad_token)
    )


if __name__ == '__main__':
    main()
