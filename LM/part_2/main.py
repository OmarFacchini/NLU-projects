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
    parser.add_argument('--optimizer', default='SGD',type=str, choices=['SGD','AdamW'], help='optimizer to use: SGD or AdamW')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--dropout', default=1, type=int, choices=[0,1], help='use dropout: 1(true) or 0(False)')
    parser.add_argument('--lr', default=0.01, type=float, help="learning rate to use")
    parser.add_argument('--regularization', default=0, choices=[0,1,2,3], type=int, help="regularization technique to use: 1: Weight Tying 2: Variational Dropout(no DropConnect) 3: Non-monotonically Triggered AvSGD")
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

    if(args.regularization == 1):
        emb_size = hid_size
    else:
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
    '''
    data_path = {'train': 'LM/part_2/dataset/PennTreeBank/ptb.train.txt',
                 'val': 'LM/part_2/dataset/PennTreeBank/ptb.valid.txt',
                 'test': 'LM/part_2/dataset/PennTreeBank/ptb.test.txt'}
    '''

    vocab_len, train_loader, val_loader, test_loader, padding = build_dataloaders(train_data_path=data_path['train'],
                                                                                  val_data_path=data_path['val'],
                                                                                  test_data_path=data_path['test'])
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=padding)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=padding, reduction='sum')

    # model with weight tying, the actuall tying is done inside the __init__ of the model
    if(args.regularization == 1):
        model = LM_LSTM(vocab_size=vocab_len, 
                    padding_index=padding, 
                    train_criterion=criterion_train, 
                    eval_criterion=criterion_eval, 
                    embedding_dim=emb_size,
                    hidden_dim=hid_size,
                    dropout=args.dropout,
                    regularization=args.regularization,
                    device=device).to(device)
        
    # model with Variational Dropout, the dropout value is fixed inside the __init__ of the model
    # could provide it as a parameter but it will complicate and create caos so i'd rather keep it fixed and change it myself
    elif(args.regularization == 2):
        model = Variational_Dropout_LM_LSTM(vocab_size=vocab_len,
                                            padding_index=padding,
                                            train_criterion=criterion_train,
                                            eval_criterion=criterion_eval,
                                            embedding_dim=emb_size,
                                            hidden_dim=hid_size,
                                            regularization=args.regularization,
                                            device=device).to(device)
        
    elif(args.regularization == 3):
        model = LM_LSTM(vocab_size=vocab_len, 
                    padding_index=padding, 
                    train_criterion=criterion_train, 
                    eval_criterion=criterion_eval, 
                    embedding_dim=emb_size,
                    hidden_dim=hid_size,
                    dropout=args.dropout,
                    regularization=3,
                    device=device).to(device)
        
        # copy of the model in which we will apply the average parameters
        # this is needed in order to not modify the original model that we want to maintain
        # we could store the original parameters and swap original and average on the same model
        # but could cause errors and is harder to understand, having an entire extra model takes up more memory
        # but is way clearer
        validation_model = LM_LSTM(vocab_size=vocab_len, 
                                   padding_index=padding, 
                                   train_criterion=criterion_train, 
                                   eval_criterion=criterion_eval, 
                                   embedding_dim=emb_size,
                                   hidden_dim=hid_size,
                                   dropout=args.dropout,
                                   regularization=3,
                                   device=device).to(device)
        
        # start by using standard SGD and after X epochs use AvSGD
        avg_start = 10 

        # clone the parameters and store them in a dictionary with name of parameter as key to maintain a live average
        # detatch as we don't want them in the computation graph
        avg_params = {name: param.clone().detach().requires_grad_(False) for name, param in model.named_parameters()}

    # standard case is LM_LSTM with no regularization(basically part 1)
    else:
        model = LM_LSTM(vocab_size=vocab_len, 
                    padding_index=padding, 
                    train_criterion=criterion_train, 
                    eval_criterion=criterion_eval, 
                    embedding_dim=emb_size,
                    hidden_dim=hid_size,
                    dropout=args.dropout,
                    regularization=0,
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


    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        # check if we are using AvSGD and if we reached the starting of averaging
        if args.regularization == 3 and epoch >= avg_start:
            loss, avg_params = train(model=model, data=train_loader, optimizer=optimizer, clip=clip, avg_params=avg_params, avg=True)
        else: # if we are either not using AvSGD or we haven't reached the starting of averaging epoch, do standard train
            loss, _ = train(model=model, data=train_loader, optimizer=optimizer, clip=clip)
        
        #could literally remove this, simply tells the model to run validation every epoch
        #could be useful if we have validation every X epoch, but need to move the progress bar update out of this if
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            # check if we are using AvSGD and if we reached the starting of averaging
            if args.regularization == 3 and epoch >= avg_start:
                with torch.no_grad():
                    # copy the average parameters in the model used for validation
                    for name, param in validation_model.named_parameters():
                        param.copy_(avg_params[name])

                    # if we have to use the averaged values
                ppl_val, loss_val, val_mode = validation(model=validation_model, data=val_loader, validation_model=True)
            else: # if not AvSGD or not in averaging epochs, use standard model
                ppl_val, loss_val, val_mode = validation(model=model, data=val_loader)


            #perplexity_list.append(ppl_val)
            #losses_val.append(np.asarray(loss_val).mean())
            pbar.set_description("PPL: %f" % ppl_val)

            if  ppl_val < best_ppl: # the lower, the better
                best_ppl = ppl_val

                # if we used the averaged model, store that
                if val_mode:
                    best_model = copy.deepcopy(validation_model).to(CPU)
                else: # else store original model
                    best_model = copy.deepcopy(model).to(CPU)

                # reset patience as we got a better model    
                patience_current = patience_fixed
            else: # decrease patience
                patience_current -= 1

            if patience_current <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean


    best_model.to(device)

    final_ppl, _, _ = validation(model=best_model, data=test_loader)
    print('Test ppl: ', final_ppl)
    torch.save(model.state_dict(), "./models/"+exp_name+".pth")
