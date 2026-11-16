## GRU

We adapted the tutorial's LSTM encoder to use PyTorch's built-in GRU. We use one layer to keep the initial model simple. The GRU reads words in order and uses its final state to represent each sentence.

We process the premise and hypothesis separately using the same GRU, starting with fresh memory for each sentence. We then concatenate their final states, premise first, so their roles remain distinct. This gives each pair a representation of size `2 * hidden_size`.

The fastText embeddings are frozen by default, but we can allow training to adjust them to compare the two approaches. This is a sort of fine-tuning. When doing this, each fold gets a fresh copy of the original embeddings. We ignore padding when reading sentences so it does not affect their final states. Optional word dropout hides whole word vectors during training and it is off by default.

## Tokenization

We use the same fastText tokenizer as in the preprocessing pipeline to split each premise and hypothesis into words. We then convert each word into its index in the pretrained fastText vocabulary, so the GRU can look up the corresponding embedding.

We tokenize the premise and hypothesis separately to preserve their roles. Since they can have different lengths, we pad the shorter sentence with the embedding model's padding token and store the result as two columns of word IDs. If a sentence is empty, we replace it with the unknown-token ID so the GRU always receives at least one token.

We apply this conversion to every example in the training, validation, and test splits, keeping the original labels and fold assignments. The tokenized datasets are saved separately to disk so we do not need to repeat tokenization during training. The original text datasets remain unchanged.