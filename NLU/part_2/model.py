import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch
    
class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, criterion_slots, criterion_intents, n_layer=1, pad_index=0, dropout=True):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        self.embedding_size = emb_size
        self.hidden_size = hid_size
        self.out_intent = out_int
        self.out_slot = out_slot
        self.vocab_len = vocab_len
        self.pad_index = pad_index
        self.use_dropout = dropout
        self.bidirection = True

        self.criterion_slots = criterion_slots
        self.criterion_intents = criterion_intents
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=self.bidirection, batch_first=True)  

        if self.bidirection: 
          self.slot_out = nn.Linear(2*hid_size, out_slot)
        else:
          self.slot_out = nn.Linear(hid_size, out_slot)


        self.intent_out = nn.Linear(hid_size, out_int)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len[idx]
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len[idx] X emb_size

        if self.use_dropout:
            utt_emb = self.dropout(utt_emb)
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.use_dropout:
          utt_encoded = self.dropout(utt_encoded)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
  

class modifiedBERT(BertPreTrainedModel):
    
    def __init__(self, config: PretrainedConfig, intents, slots):
        super(modifiedBERT, self).__init__(config)
        self.intents = intents
        self.slots = slots

        self.model = BertModel(config)

        for param in self.model.parameters():
          param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.intent_out = nn.Linear(config.hidden_size, intents)
        self.slot_out = nn.Linear(config.hidden_size, slots)

        self.init_weights()


    def forward(self, token_ids, attention_mask):
        pred = self.model(token_ids, attention_mask=attention_mask)

        sequence_out = pred[0]
        pooled_out = pred[1]

        sequence_out = self.dropout(sequence_out)
        pooled_out = self.dropout(pooled_out)

        intent_pred = self.intent_out(pooled_out)
        slot_pred = self.slot_out(sequence_out)

        return intent_pred, slot_pred
      