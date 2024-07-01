import torch
import torch.nn as nn
import torch.optim as optim
    
class LM_LSTM(nn.Module):

    def __init__(self, vocab_size, padding_index, train_criterion, eval_criterion, embedding_dim=300, hidden_dim=200, dropout=True, device='cuda'):
        super(LM_LSTM, self).__init__()

        self.hidden_layers_size = hidden_dim
        self.embedded_layer_size = embedding_dim
        self.output_size = vocab_size
        self.padding_index = padding_index
        self.number_of_layers = 1
        self.useDropout = dropout
        self.device = device
        
        self.criterion_train = train_criterion
        self.criterion_eval  = eval_criterion

        # simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(num_embeddings=self.output_size, 
                                      embedding_dim=self.embedded_layer_size, 
                                      padding_idx=self.padding_index)

        # drop some random values with probability p=0.2
        if(self.useDropout):
            self.dropout = nn.Dropout(p=0.2)

        # LSTM: apply memory RNN to an input
        # note: could add the parameter dropout, but it applies to all LSTM layers EXCEPT the last one, so I would rather have it directly outside and manipulate however I want
        # for clarity
        self.LSTM = nn.LSTM(input_size=self.embedded_layer_size,
                            hidden_size=self.hidden_layers_size,
                            num_layers=self.number_of_layers, 
                            bidirectional=False,
                            batch_first=True)
        
        if(self.useDropout):
            self.dropout2 = nn.Dropout(p=0.2)

        # linear layer to map back to the uoutput space
        self.output = nn.Linear(self.hidden_layers_size, self.output_size)
    

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.number_of_layers, batch_size, self.hidden_layers_size).zero_().to(self.device),
                  weight.new(self.number_of_layers, batch_size, self.hidden_layers_size).zero_().to(self.device))
     
        '''hidden = (torch.zeros(self.number_of_layers, batch_size, self.hidden_layers_size).to(self.device),
                  torch.zeros(self.number_of_layers, batch_size, self.hidden_layers_size).to(self.device))'''
        return hidden

    
    def forward(self, token, previous_state=None):
        embedded = self.embedding(token)

        if(self.useDropout):
            embedded = self.dropout(embedded)

        LSTM_output, hidden_layer = self.LSTM(embedded, previous_state)

        if(self.useDropout):
            LSTM_output = self.dropout2(LSTM_output)

        # note that we should be taking the last layer of the lstm, but since we have only
        # a single layer, by default it's the last one and we don't need to "filter"
        output = self.output(LSTM_output).permute(0,2,1)

        return output, hidden_layer