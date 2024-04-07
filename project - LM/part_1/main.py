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


if __name__ == "__main__":
    n_epochs = 100
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
    learning_rate = 0.01
    clip = 5
    GPU = 'cuda:0'
    CPU = 'cpu'

    # get training file
    get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.train.txt",
              data_path='./dataset/')
    
    # get validation file
    get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.valid.txt",
              data_path='./dataset/')
    
    # get test file
    get_files(data_url="https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/dataset/PennTreeBank/ptb.test.txt",
              data_path='./dataset/')


    data_path = {'train': './dataset/ptb.train.txt',
                 'val': './dataset/ptb.valid.txt',
                 'test': './dataset/ptb.test.txt'
                 }

    vocab_len, train_loader, val_loader, test_loader = build_dataloaders(train_data_path=data_path['train'],
                                                                         val_data_path=data_path['val'],
                                                                         test_data_path=data_path['test'])


    model = LM_LSTM(embedding_dim= emb_size, hidden_dim=hid_size, vocab_size=vocab_len, padding_index=0)
    model.to(GPU)
    model.init_weights()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = model.train(data=train_loader, optimizer=optimizer, clip=clip)

        #could literally remove this, simply tells the model to run validation every epoch
        #could be useful to have validation every X epoch, but need to move the progress bar update out of this if
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())

            ppl_val, loss_val = model.validation(data=val_loader)
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

    plot_results(data=perplexity_list, epochs=n_epochs, label='perplexity')
    plot_results(data=losses_val, epochs=n_epochs, label='val_loss')
    plot_results(data=losses_train, epochs=n_epochs, label='train_loss')

    best_model.to(GPU)

    final_ppl, _ = best_model.validation(data=test_loader)
    print('Test ppl: ', final_ppl)
