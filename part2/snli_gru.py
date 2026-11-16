import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    """Read both SNLI sentences and join their vectors in premise-hypothesis order.

    Adapted from the tutorial's LSTMEncoder. The GRU returns a hidden state
    without the additional cell state used by an LSTM.
    """

    def __init__(
        self,
        hidden_size,
        embedding_matrix,
        padding_idx=0, # index for padding
        freeze_embeddings=True, # if we want to fine tune embeddings, set this to false
        word_dropout=0.0,
    ):
        super().__init__()
        if not 0 <= word_dropout < 1:
            raise ValueError("word_dropout must be between 0 and 1")

        embedding_matrix = torch.as_tensor(embedding_matrix, dtype=torch.float32) # for type safety 

        # each fold needs its own starting copy if word embeddings are to be finetuned
        if not freeze_embeddings:
            embedding_matrix = embedding_matrix.clone()

        # set u[p the embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )

        # store config
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout

        # setup the GRU, its easy just built in 
        self.gru = nn.GRU(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    # properties
    @property
    def enc_dim(self):
        return 2 * self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx


    def _encode_sentence(self, x):
        """Map right-padded word IDs [batch, length] to [batch, hidden_size]."""
        valid = x.ne(self.pad_idx) # mask for non padded words
        lengths = valid.sum(dim=1) # calculate sequence lengths from pad
        if torch.any(lengths == 0): # reject empty sequences
            raise ValueError("Each sentence must contain at least one non-padding word")

        embedded = self.embedding(x)

        # optional word dropout: hide whole word vectors during training only.
        if self.training and self.word_dropout > 0:
            drop_mask = (torch.rand(x.shape, device=x.device) < self.word_dropout) & valid
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        # ignore padding so the final state corresponds to the last actual word
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    # encodes and concatenates the premises and hypotheses
    def forward(self, x):
        """Map paired word IDs [batch, length, 2] to [batch, 2 * hidden_size].

        Column 0 contains premise IDs; column 1 contains hypothesis IDs.
        Both sentences use the same GRU, each starting with fresh memory.
        """
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError("Expected paired word IDs with shape [batch, length, 2]")
        premise = self._encode_sentence(x[..., 0])
        hypothesis = self._encode_sentence(x[..., 1])
        return torch.cat((premise, hypothesis), dim=-1)

