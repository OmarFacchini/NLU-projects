# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *

from tqdm import tqdm
import copy
import math
import torch.optim as optim
import numpy as np
import argparse



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='SGD',type=str, help='optimizer to use: SGD or AdamW')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--dropout', default=1, type=int, choices=[0,1], help='use dropout: 1(true) or 0(False)')
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate to use")
    parser.add_argument('--exp_name', default='myModel', type=str, help="name of the experiment and model that will be stored")

    args = parser.parse_args()

    n_epochs = args.epochs
    patience_fixed = 5
    patience_current = patience_fixed
    losses_train = []
    losses_val = []
    perplexity_list = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))

    hid_size = 200
    emb_size = 300

    learning_rate = args.lr
    clip = 5

    GPU = "cuda:0"
    CPU = 'cpu'

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # set a seed for reproducibility of experiments
    torch.manual_seed(32)
    exp_name = args.exp_name

    data_path = {'train': 'dataset/PennTreeBank/ptb.train.txt',
                 'val': 'dataset/PennTreeBank/ptb.valid.txt',
                 'test': 'dataset/PennTreeBank/ptb.test.txt'
                 }
    
    # path for debugger
    '''data_path = {'train': 'LM/part_1/dataset/PennTreeBank/ptb.train.txt',
                 'val': 'LM/part_1/dataset/PennTreeBank/ptb.valid.txt',
                 'test': 'LM/part_1/dataset/PennTreeBank/ptb.test.txt'}'''
    

    vocab_len, train_loader, val_loader, test_loader, padding = build_dataloaders(train_data_path=data_path['train'],
                                                                                  val_data_path=data_path['val'],
                                                                                  test_data_path=data_path['test'])
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=padding)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=padding, reduction='sum')

    # standard LM_LSTM
    model = LM_LSTM(vocab_size=vocab_len, 
                    padding_index=padding, 
                    train_criterion=criterion_train, 
                    eval_criterion=criterion_eval, 
                    embedding_dim=emb_size,
                    hidden_dim=hid_size,
                    dropout=args.dropout,
                    device=device).to(device)
    model.apply(init_weights)

    # check the optimizer provided in the arguments, note that the default value is SGD
    # if optimizer is not provided, use SGD, if the provided optimizer is not one of the options, use the default SGD with lr=4
    if(args.optimizer == "AdamW"):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif(args.optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=4)

    for epoch in pbar:
        loss = train(model=model, data=train_loader, optimizer=optimizer, clip=clip)

        #could literally remove this, simply tells the model to run validation every epoch
        #could be useful if we have validation every X epoch, but need to move the progress bar update out of this if
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_val, loss_val = validation(model=model, data=val_loader)

            perplexity_list.append(ppl_val)

            losses_val.append(np.asarray(loss_val).mean())
            pbar.set_description("PPL: %f" % ppl_val)
            if  ppl_val < best_ppl: # the lower, the better
                best_ppl = ppl_val
                best_model = copy.deepcopy(model).to(CPU)

                # reset patience as we got a better model    
                patience_current = patience_fixed
            else: # decrease patience
                patience_current -= 1

            if patience_current <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean


    best_model.to(device)

    final_ppl, _ = validation(model=best_model, data=test_loader)
    print('Test ppl: ', final_ppl)
    torch.save(model.state_dict(), "./models/"+exp_name+".pth")
