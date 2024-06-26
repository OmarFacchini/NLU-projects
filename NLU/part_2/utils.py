# Add functions or classes used for data loading and preprocessing

import torch
from torch.utils.data import Dataset
from subprocess import run


class IntentsAndSlots(Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_labels(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_labels(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    

# Loading the corpus
def load_data(path):
    from json import loads
    dataset = []
    with open(path) as f:
        dataset = loads(f.read())
    return dataset


def get_files(data_url, store_path):
    # make the OS run the wget command to get the files that will be our dataset
    run(["wget", "-p", store_path, data_url])


def generate_validation_set(training_set_raw, percentage=0.1):
    from collections import Counter
    from sklearn.model_selection import train_test_split

    intents = [x['intent'] for x in training_set_raw]
    count_intents = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for idx, intent in enumerate(intents):
        # if intent occurs only once, put it in train
        if(count_intents[intent] > 1):
            inputs.append(training_set_raw[idx])
            labels.append(intent)
        else: #else put it in val
            mini_train.append(training_set_raw[idx])

    x_train, x_val, intent_train, intent_val = train_test_split(inputs, labels, test_size=percentage, random_state=42, shuffle=True, stratify=labels)

    x_train.extend(mini_train)
    train_raw = x_train
    val_raw = x_val


    ''' train_raw[0]= {'intent': 'airfare',
                        'slots': 'O O O O O O O O B-fromloc.city_name O B-toloc.city_name',
                        'utterance': 'what is the cost for these flights from baltimore to '
                                     'philadelphia'
                       }
        y_train[0] = intent of train_raw[0]
            
        val_raw[0] is same as train but for the validation set (generated one)
        y_val[0] = intent of val_raw[0]

        test_raw[0] same as the other two but for test set
        y_test[0] = intent of test_raw[0]
    
    '''

    # Intent distributions
    # print('Train:')
    # pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    # print('Dev:'), 
    # pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    # print('Test:') 
    # pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    # print('='*89)
    # # Dataset size
    # print('TRAIN size:', len(train_raw))
    # print('DEV size:', len(dev_raw))
    # print('TEST size:', len(test_raw))

    return train_raw, intent_train, val_raw, intent_val


    



    