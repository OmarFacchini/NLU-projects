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
import wandb



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='SGD',type=str, help='optimizer to use: SGD or AdamW')
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

    wandb.init(
        project="NLU_LM", 
        name=args.exp_name, 
        config={
            "learning_rate": args.lr,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "architecture": "dropout: "+ str(args.dropout),
            "dropout value": 0.2,
            "regularization": args.regularization
            })
    
    # set a seed for reproducibility of experiments
    torch.manual_seed(32)
    exp_name = args.exp_name

    # get training file, now done in the runner.sh file as i don't have a unix environment and wget is not available to me as of now
    '''get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt",
              store_path='./dataset')
    
    # get validation file
    get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
              store_path='./dataset')
    
    # get test file
    get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
              store_path='./dataset')
    '''

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
                    device=GPU).to(GPU)
        
    # model with Variational Dropout, the dropout value is fixed inside the __init__ of the model
    # could provide it as a parameter but it will complicate and create caos so i'd rather keep it fixed and change it myself
    elif(args.regularization == 2):
        model = Variational_Dropout_LM_LSTM(vocab_size=vocab_len,
                                            padding_index=padding,
                                            train_criterion=criterion_train,
                                            eval_criterion=criterion_eval,
                                            embedding_dim=emb_size,
                                            hidden_dim=hid_size,
                                            device=GPU).to(GPU)
        
    # standard case is LM_LSTM with no regularization(basically part 1)
    # note: for args.regularization == 3 we have a special case where we use the basic model and use a custom optimizer defined below
    else:
        model = LM_LSTM(vocab_size=vocab_len, 
                    padding_index=padding, 
                    train_criterion=criterion_train, 
                    eval_criterion=criterion_eval, 
                    embedding_dim=emb_size,
                    hidden_dim=hid_size,
                    dropout=args.dropout,
                    regularization=0,
                    device=GPU).to(GPU)

    model.apply(init_weights)

    # check the optimizer provided in the arguments, note that the default value is SGD
    # if optimizer is not provided, use SGD, if the provided optimizer is not one of the options, use the default SGD
    if(args.optimizer == "AdamW"):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif(args.optimizer == "SGD"):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # this is weird but here we want to change optimizer based on the regularization we want, so it fits here rather than when defining models
    # use Non-monotonically Triggered AvSGD as optimizer
    # note: important to have size of window > patience unless you always want to consider the full window for the update which would defeat the whole purpose
    elif(args.regularization == 3):
        optimizer = My_AvSGD(params=model.parameters(), lr=learning_rate, validation_window=10, patience=5, threshold=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=4)


    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train(model=model, data=train_loader, optimizer=optimizer, clip=clip)

        #could literally remove this, simply tells the model to run validation every epoch
        #could be useful to have validation every X epoch, but need to move the progress bar update out of this if
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            #ppl_val, loss_val = model.validation(data=val_loader)
            ppl_val, loss_val = validation(model=model, data=val_loader)
            perplexity_list.append(ppl_val)

            losses_val.append(np.asarray(loss_val).mean())
            pbar.set_description("PPL: %f" % ppl_val)
            if  ppl_val < best_ppl: # the lower, the better
                best_ppl = ppl_val
                best_model = copy.deepcopy(model).to(CPU)
                patience_current = patience_fixed
            else:
                patience_current -= 1

            if patience_current <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

            wandb.log({"train_loss": loss, "validation_loss": loss_val}, step=epoch)
            wandb.log({"perplexity": ppl_val}, step=epoch)


    '''plot_results(data=perplexity_list, epochs=n_epochs, label='perplexity')
    plot_results(data=losses_val, epochs=n_epochs, label='val_loss')
    plot_results(data=losses_train, epochs=n_epochs, label='train_loss')'''

    best_model.to(GPU)

    final_ppl, _ = validation(model=best_model, data=val_loader)
    print('Test ppl: ', final_ppl)
    torch.save(model.state_dict(), "./models/"+exp_name+".pth")
