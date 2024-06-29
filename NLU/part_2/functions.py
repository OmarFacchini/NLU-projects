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
    device='cpu'
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
    origin_utt, _ = merge(new_item['original_utterance_ids'])
    
    origin_utt.to(device)
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item['original_utterances_ids'] = origin_utt
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item



def build_dataloaders(train_raw, val_raw, test_raw, lang, tokenizer):

    train_dataset = utils.IntentsAndSlots(train_raw, lang, tokenizer=tokenizer, myType='train')
    val_dataset = utils.IntentsAndSlots(val_raw, lang, tokenizer=tokenizer, myType='val')
    test_dataset = utils.IntentsAndSlots(test_raw, lang, tokenizer=tokenizer, myType='test')

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


## ====================================== model related functions ========================================== ##
def train(model, data, optimizer, criterion_slots, criterion_intents, clip=5, device='cuda:0'):
        device = 'cpu'
        model.train()
        loss = 0
        total_loss = 0

        for sample in data:

            # input_ids.shape = batch_size * max_len
            input_ids = sample['utterances'].to(device)

            # attention_mask.shape = batch_size * max_len
            attention_mask = torch.stack(sample['attention_mask']).to(device)

            # intents.shape = batch_size
            intents = sample['intents'].to(device)

            # slots.shape = batch_size * max_len
            slots = sample['y_slots'].to(device)

            optimizer.zero_grad() # Zeroing the gradient

            # intent_pred.shape = batch_size * number_of_intents(len(total_intents))
            # slot_pred.shape = batch_size * max_len * number_of_slots(129????)
            intent_pred, slot_pred = model(token_ids=input_ids, attention_mask=attention_mask)

            #loss_slot = criterion_slots(slot_pred.view(-1,sample['slots_len']), sample['y_slots'])
            loss_slot = criterion_slots(slot_pred.view(-1, model.slots), slots.view(-1))

            #loss_intent = criterion_intents(intent_pred, sample['intents'])
            loss_intent = criterion_intents(intent_pred, intents)

            loss = loss_intent + loss_slot

            total_loss += loss.item()
            
            loss.backward() # Compute the gradient

            # clip the gradient to avoid explosioning gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Update the weights

        return total_loss/len(data)   


def validation(model, data, lang, criterion_slots, criterion_intents, device='cuda:0'):
    device = 'cpu'
    model.eval()

    with torch.no_grad():
        total_loss = 0
        ref_intents = []
        hyp_intents = []
        
        ref_slots = []
        hyp_slots = []
        
        for sample in data:

            # input_ids.shape = batch_size * max_len
            input_ids = sample['utterances'].to(device)
            original_utt_ids = sample['original_utterances_ids']

            # attention_mask.shape = batch_size * max_len
            attention_mask = torch.stack(sample['attention_mask']).to(device)

            # intents.shape = batch_size
            intents = sample['intents'].to(device)

             # slots.shape = batch_size * max_len
            slots = sample['y_slots'].to(device)

            # intent_pred.shape = batch_size * number_of_intents(len(total_intents))
            # slot_pred.shape = batch_size * max_len * number_of_slots(130)
            intent_pred, slot_pred = model(token_ids=input_ids, attention_mask=attention_mask)


            loss_slot = criterion_slots(slot_pred.view(-1,model.slots), slots.view(-1))
            loss_intent = criterion_intents(intent_pred, intents)

            loss = loss_intent + loss_slot

            total_loss += loss.item()

            # mapping from ID to intent label of the prediction
            # torch.argmax(intent_pred, dim=1).shape = batch_size
            # len(predicted_intents) = 64
            predicted_intents = [lang.id2intent[x] for x in torch.argmax(intent_pred, dim=1).tolist()] 

            # map from ID to intent label the original intents
            # len(real_intents) = 64
            real_intents = [lang.id2intent[x] for x in intents.tolist()]

            # global list of real intents
            # ref = reference
            ref_intents.extend(real_intents)
            
            # global list of predicted intents
            # hyp = hypothetical
            hyp_intents.extend(predicted_intents)

            # predicted_slots.shape = batch_size * max_len
            predicted_slots = torch.argmax(slot_pred, dim=2)

            for idx, seq in enumerate(predicted_slots):
                # is max_len, if all samples are padded to same len
                # otherwise is length of the sample
                length = sample['slots_len'].tolist()[idx]

                # note that this doesn't properly get the utterance as we are using the tokenizer
                # of the model rather than our own mapping
                utt_ids = original_utt_ids[idx][:length].tolist()

                # take real slots ids
                # len(real_slots_ids) = max_len
                # gt_ids
                real_slots_ids = slots[idx].tolist()

                # convert slots ids into the actual labels of the slot
                # len(real_slots_labels) = max_len
                # gt_slots
                real_slots_labels = [lang.id2slot[elem] for elem in real_slots_ids[:length]]

                #utterance = [elem for elem in utt_ids]

                # get predicted_slots ids for the sample
                # len(to_decode) = max_len
                to_decode = seq[:length].tolist()

                # global list of real slots
                # len([]) = batch_size
                # len([()]) = max_len
                # [(utterance, slot_label), ...]
                ref_slots.append([(utt_ids[id_el], elem) for id_el, elem in enumerate(real_slots_labels)])

                tmp_seq = []
                # convert predicted slots ids into the actual labels of the slot
                # len(predicted_slots_labels) = max_len
                # predicted_slots_labels = [lang.id2slot[elem] for elem in to_decode[:length]]
                # instead of doing that, we loop on the decode and append a tuple to global list of predicted_slots
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utt_ids[id_el], lang.id2slot[elem]))
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
