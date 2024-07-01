# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

#what should i put in model.py then???

from functools import partial
from torch import LongTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import math
from torch import nn

import utils

## ====================================== data load related functions ========================================== ##
class Lang():
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}


    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output
    

def collate_fn(data, pad_token, device='cuda:0'):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item



def build_dataloaders(train_data_path, val_data_path, test_data_path):
    train_raw = utils.read_file(train_data_path)
    dev_raw = utils.read_file(val_data_path)
    test_raw = utils.read_file(test_data_path)

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_length = len(lang.word2id)
    padding = lang.word2id["<pad>"]

    train_dataset = utils.PennTreeBank(train_raw, lang)
    dev_dataset = utils.PennTreeBank(dev_raw, lang)
    test_dataset = utils.PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    return vocab_length, train_loader, dev_loader, test_loader, padding


## ====================================== model related functions ========================================== ##
def train(model, data, optimizer, clip=5):
        model.train()
        loss = 0
        total_loss = 0
        number_of_tokens = []

        for sample in data:
            hidden = model.init_hidden(sample['source'].size(0))

            optimizer.zero_grad() # Zeroing the gradient

            # get predictions
            output, hidden = model(sample['source'], hidden)


            loss = model.criterion_train(output, sample['target'])
            total_loss += loss.item() * sample["number_tokens"]
            
            number_of_tokens.append(sample["number_tokens"])
            loss.backward() # Compute the gradient
            # clip the gradient to avoid explosioning gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Update the weights

        return total_loss/sum(number_of_tokens)   


def validation(model, data):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        number_of_tokens = []
        
        for sample in data:
            hidden = model.init_hidden(sample['source'].size(0))
            
            output, hidden = model(sample['source'], hidden)
            #output = model(sample['source'])

            #could remove loss and directly edit the total_loss but this looks cleaner and clearer
            loss = model.criterion_eval(output, sample['target'])
            total_loss += loss.item()
            
            number_of_tokens.append(sample["number_tokens"])   

    # calculate perplexity and averaged loss that will be returned as measures for performance
    perplexity = math.exp(total_loss / sum(number_of_tokens))
    average_loss = total_loss/sum(number_of_tokens)

    return perplexity, average_loss

def init_weights(mat):
        for m in mat.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                    elif 'weight_hh' in name:
                        for idx in range(4):
                            mul = param.shape[0]//4
                            torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            else:
                if type(m) in [nn.Linear]:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)

## ====================================== extra utility functions ========================================== ##

def plot_results(data, epochs, label):
    epochs_list = range(1,epochs+1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs_list, data, label=label)
    plt.xlabel('epochs')
    plt.ylabel(label)
    plt.title(label + ' evolution')
    plt.legend()

    plt.savefig(label+'.png')
