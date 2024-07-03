# Description of the task

In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally. If adding a modification decreases the performance, you can remove it and move forward with the others. However, in the report, you have to provide and comment on this unsuccessful experiment. For each of your experiments, you have to print the performance expressed with Perplexity (PPL).
One of the important tasks of training a neural network is hyperparameter optimization. Thus, you have to play with the hyperparameters to minimise the PPL and thus print the results achieved with the best configuration (in particular the learning rate).

    The steps to implement:
    1. Replace RNN with a Long-Short Term Memory(LSTM) network
    2. Add two dropout layers:
        - one after the embedding layer
        - one before the last linear layer
    3. replace SGD with AdamW

# Requirements

As per path in the code, the dataset is required to be present in the folder `part_1`, to get that you can simply run this code in the shell, if you are on a platform that supports the command wget.

``` 
wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt
wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt
wget -P dataset/PennTreeBank https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt 
```

Otherwise simply go to the link and download it by hand, make sure the path you store in this folder is `/dataset/PennTreeBank/`.

As per the library requirements, the file `requirements.txt` should suffice.

# Usage

As a parser was used, there are two options to run the code.

The first and easiest one is utilizing the file `runner.sh` where the parameters can be modified at will. Assuming you are in the correct folder, meaning the file .sh is directly in your folder, to run the code simply do 
``` .\runner.sh ```

The other option is to run in the shell the command:
``` python main.py --optimizer AdamW --epochs 100 --dropout 1 --lr 0.005 --exp_name SGD001 ```

Also ensure there is a folder named `models` in the same folder as the .sh as it's where the best model will be stored at the end of the testing.