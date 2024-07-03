# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import copy
import math
import torch.optim as optim
import numpy as np
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--dropout', default=1, type=int, choices=[0,1], help='use dropout: 1(true) or 0(False)')
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate to use")
    parser.add_argument('--exp_name', default='myModel', type=str, help="name of the experiment and model that will be stored")

    args = parser.parse_args()

    n_epochs = args.epochs
    patience_fixed = 5
    current_patience = patience_fixed

    losses_train = []
    losses_val = []
    sampled_epochs = []

    best_model = None
    pbar = tqdm(range(1,n_epochs))

    GPU = "cuda:0"
    CPU = 'cpu'

    
    # set a seed for reproducibility of experiments
    torch.manual_seed(32)
    exp_name = args.exp_name

    
    data_path = {'train': './dataset/ATIS/train.json',
                 'test': './dataset/ATIS/test.json'
                }
    
    
    #DATA PATH FOR DEBUGGER
    '''
    data_path = {'train': 'NLU/dataset/ATIS/train.json',
                 'test': 'NLU/dataset/ATIS/test.json'
                }
    '''

    tmp_train_raw_data = load_data(data_path['train'])
    test_raw_data = load_data(data_path['test'])
    print('Train samples:', len(tmp_train_raw_data))
    print('Test samples:', len(test_raw_data))


    # tmp_train_raw_data[0] is: {'intent': -
    #                            'slots': -
    #                            'utterance': -
    #                           }
    
    # build the validation set
    # get the trimmed train set with its intent, and same for validation
    # need to return the train set too as we have a subset of the original (tmp_train_raw_data - train_raw = val_raw)
    train_raw_data, intent_train, val_raw_data, intent_val = generate_validation_set(training_set_raw=tmp_train_raw_data, percentage=0.1)

    intent_test, words, corpus, slots, total_intents = get_dataset_informations(train_raw_data=train_raw_data, 
                                                                                val_raw_data=val_raw_data, 
                                                                                test_raw_data=test_raw_data)

    lang = Lang(words, total_intents, slots, cutoff=0)

    slots.add('pad')

    PAD_TOKEN = 0

    lr = args.lr
    clip = 5 # Clip the gradient

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    model_name = 'bert-base-uncased'
    max_token_len = 50

    model = modifiedBERT.from_pretrained(model_name, intents=len(total_intents), slots=len(slots)).to(GPU)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    train_loader, val_loader, test_loader = build_dataloaders(train_raw=train_raw_data, 
                                                              val_raw=val_raw_data, 
                                                              test_raw=test_raw_data, 
                                                              lang=lang,
                                                              tokenizer=tokenizer)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_f1 = 0
    best_model = model


    for epoch in pbar:
        loss = train(model=model, data=train_loader, optimizer=optimizer, clip=clip, criterion_slots=criterion_slots, criterion_intents=criterion_intents)
        loss = 0

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            results_val, intent_res, loss_val = validation(model=model, data=train_loader, lang=lang, criterion_slots=criterion_slots, criterion_intents=criterion_intents, tokenizer=tokenizer)
            losses_val.append(np.asarray(loss_val).mean())
        
            f1 = results_val['total']['f']

            pbar.set_description("f1: %f" %f1)

            if f1 > best_f1:
                best_f1 = f1
                current_patience = patience_fixed
                best_model = copy.deepcopy(model).to(CPU)
            else:
                current_patience -= 1

            if current_patience <= 0:
                break
    
    best_model.cuda()
    results_test, intent_test, _ = validation(model=best_model, data=test_loader, lang=lang, criterion_slots=criterion_slots, criterion_intents=criterion_intents, tokenizer=tokenizer)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    torch.save(best_model.state_dict(), "./models/"+exp_name+".pth")

