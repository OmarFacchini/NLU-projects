# Description of the task

Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
You can refer to this paper to have a better understanding of how to implement this: https://arxiv.org/abs/1902.10909. In this, one of the challenges of this is to handle the sub-tokenization issue.

*Note*: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.


The models that you can experiment with are [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll


# Requirements

As per path in the code, the dataset is required to be present in the folder `part_2`, to get that you can simply run this code in the shell, if you are on a platform that supports the command wget.

``` 
wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json
```

Otherwise simply go to the link and download it by hand, make sure the path you store in this folder is `dataset\ATIS\`.

Also take a note to run or download this:
```
wget https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py 
``` 

As it's what will be used to evaluate the models.

As per the library requirements, the file `requirements.txt` should suffice.

# Usage

As a parser was used, there are two options to run the code.

The first and easiest one is utilizing the file `runner.sh` where the parameters can be modified at will. Assuming you are in the correct folder, meaning the file .sh is directly in your folder, to run the code simply do 
``` .\runner.sh ```

The other option is to run in the shell the command:
``` python main.py --epochs 100 --dropout 0 --lr 0.1 --exp_name test ```

Also ensure there is a folder named `models` in the same folder as the .sh as it's where the best model will be stored at the end of the testing.
