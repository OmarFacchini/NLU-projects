# Description

In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment. For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
One of the important tasks of training a neural network is hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular the learning rate).

    The steps to implement:
    1. Replace RNN with a Long-Short Term Memory(LSTM) network
    2. Add two dropout layers:
        - one after the embedding layer
        - one before the last linear layer
    3. replace SGD with AdamW