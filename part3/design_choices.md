# Group 37 assignment 3 design choices and answers
## Questions and answers
1. Framing the task as a 0-shot and few-shot prompting:
    * What LLM are you using and why?
    * How do you frame classification as prompting? Do you prompt the model to generate the label or did you formulate classification as multiple-choice question, or else?
    * Does the instruction affect classification performance? 
    * Do the number, order or particular choice of the examples/demonstrations affect classification performance?
    * Do sampling parameters used in generation affect classification performance?
    * These are some hyperparameters that need to be considered when employing an LLM for the task. You could even consider to extend your experiments to a different family of LLMs, and see whether they perform better (however, be considerate of the resources available to you; stick to a model of a similar size in terms of model parameters). 
2. Fine tuning:
    * What approach did you use and why? Can you sketch, briefly and intuitively, how this technique makes the fine tuning problem more computationally approachable with the resources you have access to?
    * What hyperparameters did you investigate and why? What impacts did you observe in resource consumption? What were the effects in classification performance? What tradeoffs did you settle for and why?
3. Comparisons:
    * How do the different LLM-based approaches compare? How are you keeping an eye on experimental uncertainty? 
    * How about comparisons to the techniques from Projects 1 and 2?
    * You now have a lot of results to consider, think of appropriate ways to display them and support meaningful discussions.
    * Any serious discussion should also take into account the resource perspective (tradeoffs between improved classification and increased demand for resources).
4. In most instances, LLM-based approaches should be motivated by much more than some improved accuracy or F1. To even begin to make a case for the costs and broader impact of LLMs, we need to at a minimum be able to show that these LLM-based approaches achieve something that earlier approaches were not likely to achieve. What could that be?
    * In the tutorial we investigate domain shift. You should investigate that too. How did you obtain out-of-domain data (if you cannot find relevant OOD data, feel free to construct some, even a small sample, but give it some thought, motivate your choices and be critical about the quality of your data)? What did you observe in terms of classification performance?
    * We also investigated the case of robustness to pharaphrastic variation. You should investigate that too. What paraphrasing approach did you use for your task? Does it seem to work well / did you attempt to evaluate it? What was the impact of paraphrastic variation in your classifiers?
    * Is there another way in which you could stress test your models further? Motivate another form of 'stress/robustness test' (another form of domain shift, or some other form of unexpected variation, or even an 'adversarial attack') and see how/if your approach holds up. Why did you choose this specific test? How did you operationalise it? And what did you observe?
5. What are some ideas (other than collecting training data for the specific 'attack') to improve performance, or at least avoid leading users to errors, in (4)? For one of these ideas, can you articulate a brief, but convincing, rationale for its feasibility and for why one should expect it to be effective?