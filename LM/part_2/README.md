# Description of the task

For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the model achieved from `part 1`, apply the following regularization techniques:
- Weight Tying
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).

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
``` python main.py --optimizer AdamW --epochs 100 --dropout 1 --lr 0.005 --regularization 1 --exp_name AdamropWT0005 ```

where regularization can be:
- 1: Weight Tying
- 2: Variational Dropout
- 3: Average SGD

Also ensure there is a folder named `models` in the same folder as the .sh as it's where the best model will be stored at the end of the testing.