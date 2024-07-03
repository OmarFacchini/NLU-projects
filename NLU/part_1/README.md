# Description of the task

As for LM project, you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance. 

Modify the baseline architecture Model IAS by:
- Adding bidirectionality
- Adding dropout layer

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll

# Requirements

As per path in the code, the dataset is required to be present in the folder `part_1`, to get that you can simply run this code in the shell, if you are on a platform that supports the command wget.

``` 
wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json
```

Otherwise simply go to the link and download it by hand, make sure the path you store in this folder is `dataset\ATIS\`.

Also take a note to run or download this:
```
wget https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py 
``` 

as it's what will be used to evaluate the models.

As per the library requirements, the file `requirements.txt` should suffice.

# Usage

As a parser was used, there are two options to run the code.

The first and easiest one is utilizing the file `runner.sh` where the parameters can be modified at will. Assuming you are in the correct folder, meaning the file .sh is directly in your folder, to run the code simply do 
``` .\runner.sh ```

The other option is to run in the shell the command:
``` python main.py --epochs 100 --dropout 0 --lr 0.1 --exp_name NLU-AdamBidir01 ```

Also ensure there is a folder named `models` in the same folder as the .sh as it's where the best model will be stored at the end of the testing.