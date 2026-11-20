## GRU

We adapted the tutorial's LSTM encoder to use PyTorch's built-in GRU. We use one layer to keep the initial model simple. The GRU reads words in order and uses its final state to represent each sentence.

We process the premise and hypothesis separately using the same GRU, starting with fresh memory for each sentence. We then concatenate their final states, premise first, so their roles remain distinct. This gives each pair a representation of size `2 * hidden_size`.

The fastText embeddings are frozen by default, but we can allow training to adjust them to compare the two approaches. This is a sort of fine-tuning. When doing this, each fold gets a fresh copy of the original embeddings. We ignore padding when reading sentences so it does not affect their final states. Optional word dropout hides whole word vectors during training and it is off by default.

## Tokenization

We use the same fastText tokenizer as in the preprocessing pipeline to split each premise and hypothesis into words. We then convert each word into its index in the pretrained fastText vocabulary, so the GRU can look up the corresponding embedding.

We tokenize the premise and hypothesis separately to preserve their roles. Since they can have different lengths, we pad the shorter sentence with the embedding model's padding token and store the result as two columns of word IDs. If a sentence is empty, we replace it with the unknown-token ID so the GRU always receives at least one token.

We apply this conversion to every example in the training, validation, and test splits, keeping the original labels and fold assignments. The tokenized datasets are saved separately to disk so we do not need to repeat tokenization during training. The original text datasets remain unchanged.

## SNLI Architectures

We reuse the tutorial's cross-validation pipeline for training and evaluation. Since our token IDs are already cached, we provide a simple tokenizer wrapper that returns them directly. We use the five existing stratified folds to keep experiments comparable.

We use a GRU with hidden size 128 and the tutorial's softmax regressor with two hidden layers of size 50 and dropout of 0.1. We keep AdamW with a learning rate of `1e-3`, weight decay of `1e-4`, and batch size 64, training for 5 epochs by default.

We restore predictions to their original dataset order and save them alongside labels, fold assignments, and accuracy metrics for later analysis.


## Questions and Answers

1. Find a word embedding model that is appropriate for your dataset. Think about the language, the kind of text you are working with. Find evidence that your choice is a good one: 
    * Can you tokenize your texts in a way coherent with the definition of a token in the embedding model?
    * Does the model's vocabulary give enough coverage to the words in your texts? etc.



2. Explore what your chosen word embedding model captures. Some things you might consider: 
    * Is it good at representing interesting similarities (re to your choice of task)? J
    * Have you found task-specific evidence that it's better than some other reasonable alternative?
    * Do you know about limitations of this model?
    * Any limitations that you find particularly troubling for your task? 



3. Experiment with the embed-and-pool approach.  
    * What design choices have meaningful impact in classification performance (eg, embedding size, architecture design, optimisation parameters), if any?
    * Is this better than what you could achieve last week? In other words, did pretrained embeddings add to your design?


    
4. Experiment with a recurrent encoder (GRU or LSTM). 
    * What design choices have meaningful impact in classification performance, if any?
    * Did you investigate the impact of data size, embedding parameters (trained or not, fixed or trainable, etc)?
    * Did you struggle with over-fitting?
    * How far can this approach go? Are you able to improve meaningfully on embed-and-pool?



5. We make decisions under uncertainty all the time, are you gathering enough information to substantiate your choices? Various things can lead to small differences in results, and these differences are not always consistent. Here are some things to think about:
    * How did you structure you hyperparameter search?
    * Are you keeping an eye on experimental variance?


    
6. Experiment with error analysis and try to discover what characteristics of input texts are predictive of performance differences across major design choices. 
    * What analysis features did you come up with?
    * Which ones did you manage to test and how did that go?
    * What would you prioritise in the future in order to improve your systems?