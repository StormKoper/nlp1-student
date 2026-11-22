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

        *We have tokenized the texts using the exact tokenizer our embedding model (fastText) was trained on. This is the Europarl tokenizer packaged through Sacremoses.*

    * Does the model's vocabulary give enough coverage to the words in your texts? etc.

        *With the matching tokenizer, the vocabulary of the chosen embedding model covers over 99.9% of the tokens in the text.*


2. Explore what your chosen word embedding model captures. Some things you might consider: 
    * Is it good at representing interesting similarities (re to your choice of task)? 

        *FastText captures a few interesting similarities that are useful for SNLI. The two main ones are semantic and syntactic similarity. Semantic similarity refers to the fact that words in the same category, like dog, cat, and animal receive similar embeddings. This is helpful for SNLI because it allows the model to learn that sentences can entail one another without sharing exactly the same words. Like "a dog is running" as a premise can relate to "an animal is outside" as a hypothesis, and the embeddings help to facilitate the forming of this pattern. Syntactic similarity refers to the fact that some words that play similar grammatical roles, like quickly and slowly or walk and run, will have related embeddings. This is helpful for the SNLI dataset, especially to help distinguish when a label should be neutral or not, based on such relations between premise and hypothesis tokens. The same principle applies in the semantic setting. It is likely that the model can learn that non-similar embedding vectors in the premise and hypothesis more often correspond to a neutral label.*

    * Have you found task-specific evidence that it's better than some other reasonable alternative?

        *We haven’t established that fastText is better for SNLI. The theoretical advantage over a TF-IDF bag of tokens is that fastText can give related words similar vectors. That could help the model recognize entailment when the premise and hypothesis describe the same scene using different words. However, the same property can also make contradictions harder to spot, since words like standing and sitting often occur in similar contexts. We’d expect the clearest benefit on examples where recognizing a paraphrase matters.* 

    * Do you know about limitations of this model? Any limitations that you find particularly troubling for your task? 

        *Yes. One limitation is that embeddings are context-independent. This is especially an issue for homonymic words like "bank" of a river or "bank" where money is. They have identical embeddings. Another issue, related to the previous commentary on semantic and syntactic similarity, is that logical association is not preserved. For example hot and cold may have similar embeddings despite the fact that they are antonyms. Meanwhile hot and warm are nearly synonyms, but would also have the same embedding benefit. This is why, despite being a better starting point for SNLI, there is still a lot that our models have to learn to use these embeddings.*

3. Experiment with the embed-and-pool approach.  
    * What design choices have meaningful impact in classification performance (eg, embedding size, architecture design, optimisation parameters), if any?

    *Our implementation averaged the frozen fastText vectors for the premise and hypothesis separately, excluding padding for the purpose of this aggregation, and then concatenated them. This preserves the role of premise and hypothesis, but discards word order. The implementation supports max, mean, and sum pooling. Of these choices, the pooling type is the most significant design choice, along with the choice to preserve premise and hypothesis.* 

    * Is this better than what you could achieve last week? In other words, did pretrained embeddings add to your design?

    *No. On the first SNLI validation fold (109,874 examples), mean pooling with pretrained fastText embeddings reached 58.51% accuracy, compared with 70.55% for last week’s best TF-IDF MLP on that same fold. That’s 12.04 percentage points lower. This comparison covers one fold. The models also used different classifiers. Last week’s MLP had one hidden layer of 128 units, while embed-and-pool used two hidden layers of 50 units each. The score gap therefore reflects the full model changes, but it is reasonable to assume that even if we used the same classification head last weeks would still perform better, because the percent difference is so large. The pretrained embeddings added a way to represent lexical similarity. In theory, that helps when a premise mentions a dog and the hypothesis calls it an animal. But averaging can dilute decisive words such as no, which TF-IDF keeps as a distinct feature. That trade-off may partly explain why embed-and-pool performed worse.*
    

    
4. Experiment with a recurrent encoder (GRU or LSTM). 
    * What design choices have meaningful impact in classification performance, if any?

        *Due to limited compute, we only tested the impact of the word dropout hyperparameter. Increasing this value reduced model performance.*

    * Did you investigate the impact of data size, embedding parameters (trained or not, fixed or trainable, etc)?

        *No.*

    * Did you struggle with over-fitting?

        *No. With our limited compute budget and the cross-validation, we only trained the model for a limited number of epochs (5 for regular training and 3 for hyperparameter search). With this limited training configuration, we observed no divergence of the training and validation metrics, taking into account that the word dropout only occurs in training and thus disproportionately favors the validation metrics. Additionally, with freezing the embedding model and choosing a single-layer GRU with a hidden dimensionality of 128 trained on this corpus of half a million examples, the risk of overfitting is already small.*

    * How far can this approach go? Are you able to improve meaningfully on embed-and-pool?

        *Yes.*


5. We make decisions under uncertainty all the time, are you gathering enough information to substantiate your choices? Various things can lead to small differences in results, and these differences are not always consistent. Here are some things to think about:
    * How did you structure you hyperparameter search?

        *We tested GRU word dropout rates of 0.0, 0.1, and 0.2, using 0.0 as the baseline. We trained each setting for three epochs on the same five stratified SNLI folds. We kept the fastText embeddings frozen and the classifier dropout at 0.1. For each fold, we reported accuracy from the epoch with the lowest validation loss.*

    * Are you keeping an eye on experimental variance?
        
        *Yes. We kept track of validation accuracy on each of the five folds and reported the mean and standard deviation for each dropout setting. The means were 72.44%, 71.29%, and 69.65%, with standard deviations of 0.15, 0.10, and 0.20 percentage points, respectively. This shows the result was consistent across folds. We only used one training seed, so variation across seeds is still unknown.*


6. Experiment with error analysis and try to discover what characteristics of input texts are predictive of performance differences across major design choices. 
    * What analysis features did you come up with?
    * Which ones did you manage to test and how did that go?
    * What would you prioritise in the future in order to improve your systems?