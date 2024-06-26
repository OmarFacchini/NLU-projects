# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

#what should i put in model.py then???
import matplotlib.pyplot as plt
import torch
import math

from functools import partial
from torch import LongTensor
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import classification_report

import utils
from conll import evaluate


## ====================================== data load related functions ========================================== ##
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.pad_token = 0
        self.word2id = self.words2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.label2id(slots)
        self.intent2id = self.label2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def words2id(self, elements, cutoff=None, unk=True):
        from collections import Counter
        vocab = {'pad': self.pad_token}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def label2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = self.pad_token
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    

def collate_fn(data, pad_token=0, device='cuda:0'):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item



def build_dataloaders(train_raw, val_raw, test_raw, lang):

    train_dataset = utils.IntentsAndSlots(train_raw, lang)
    val_dataset = utils.IntentsAndSlots(val_raw, lang)
    test_dataset = utils.IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


## ====================================== model related functions ========================================== ##
def train(model, data, optimizer, clip=5):
        model.train()
        loss = 0
        total_loss = 0

        for sample in data:

            optimizer.zero_grad() # Zeroing the gradient

            slots, intent = model(sample['utterances'], sample['slots_len'])

            loss_intent = model.criterion_intents(intent, sample['intents'])
            loss_slot = model.criterion_slots(slots, sample['y_slots'])

            loss = loss_intent + loss_slot

            total_loss += loss.item()
            
            loss.backward() # Compute the gradient

            # clip the gradient to avoid explosioning gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Update the weights

        return total_loss/len(data)   


def validation(model, data, lang):
    model.eval()

    with torch.no_grad():
        total_loss = 0
        ref_intents = []
        hyp_intents = []
        
        ref_slots = []
        hyp_slots = []
        
        for sample in data:
            
            slots, intents = model(sample['utterances'], sample['slots_len'])

            loss_intent = model.criterion_intents(intents, sample['intents'])
            loss_slot = model.criterion_slots(slots, sample['y_slots'])

            loss = loss_intent + loss_slot

            total_loss += loss.item()

            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]

            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)

            for idx, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[idx]

                utt_ids = sample['utterance'][idx][:length].tolist()
                gt_ids = sample['y_slots'][idx].tolist()

                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]

                to_decode = seq[:length].tolist()

                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
        
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as exception:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", exception)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)       

    return results, report_intent, total_loss/len(data)



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


def get_dataset_informations(train_raw_data, val_raw_data, test_raw_data):
    # want to have directly the intents of the test set too, just like we did for validation and training
    intent_test = [x['intent'] for x in test_raw_data]

    # list of all words in the train set (list as we want to compute frequency of each word too)
    words = sum([x['utterance'].split() for x in train_raw_data], [])

    # list of dictionaries, [{'utterance': 'x', 'slots':'x', 'intent':'airfare'}]
    corpus = train_raw_data + val_raw_data + test_raw_data

    # set slots eg: {'I-cost_relative, 'B-arrive_time.time', 'B-return_date.day_name', ...}
    slots = set(sum([line['slots'].split() for line in corpus],[]))

    # set of all the intents in the corpus
    total_intents = set([line['intent'] for line in corpus])

    return intent_test, words, corpus, slots, total_intents
