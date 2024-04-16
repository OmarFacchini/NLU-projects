import torch
import torch.nn as nn
import torch.optim as optim
    
class LM_LSTM(nn.Module):

    def __init__(self, vocab_size, padding_index, train_criterion, eval_criterion, embedding_dim=300, hidden_dim=200, dropout=True, regularization=0, device='cuda'):
        super(LM_LSTM, self).__init__()

        self.hidden_layers_size = hidden_dim
        self.embedded_layer_size = embedding_dim
        self.output_size = vocab_size
        self.padding_index = padding_index
        self.number_of_layers = 1
        self.useDropout = dropout
        self.regularization = regularization
        self.device = device
        
        self.criterion_train = train_criterion
        self.criterion_eval  = eval_criterion

        # simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(num_embeddings=self.output_size, 
                                      embedding_dim=self.embedded_layer_size, 
                                      padding_idx=self.padding_index)

        # drop some random values with probability p
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

        # 1: Weight Tying
        # 2: Variational Dropout (no DropConnect)
        # 3: Non-monotonically Triggered AvSGD
        if(self.regularization == 1):
            self.output.weight = self.embedding.weight

        elif(self.regularization == 2):
            # apply variational dropout
            print("not implemented yet")
            return
        
        elif(self.regularization == 3):
            print("not implemented yet")
            return
            # apply Non-monotonically Triggered AvSGD


    

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
    

# a lot of doubts on this, not really clear difference between dropConnect and variational dropout
class Variational_Dropout_LM_LSTM(nn.Module):
    def __init__(self, vocab_size, padding_index, train_criterion, eval_criterion, embedding_dim=300, hidden_dim=200, device='cuda'):
        super(Variational_Dropout_LM_LSTM, self).__init__()

        self.hidden_layers_size = hidden_dim
        self.embedded_layer_size = embedding_dim
        self.output_size = vocab_size
        self.padding_index = padding_index
        self.number_of_layers = 1
        self.dropout = 0.2
        self.device = device
        
        self.criterion_train = train_criterion
        self.criterion_eval  = eval_criterion

        # simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(num_embeddings=self.output_size, 
                                      embedding_dim=self.embedded_layer_size, 
                                      padding_idx=self.padding_index)

        # LSTM: apply memory RNN to an input
        # note: could add the parameter dropout, but it applies to all LSTM layers EXCEPT the last one, so I would rather have it directly outside and manipulate however I want
        # for clarity
        self.Var_LSTM = nn.LSTM(input_size=self.embedded_layer_size,
                            hidden_size=self.hidden_layers_size,
                            num_layers=self.number_of_layers, 
                            bidirectional=False,
                            batch_first=True)

        # linear layer to map back to the uoutput space
        self.output = nn.Linear(self.hidden_layers_size, self.output_size)

    def calculate_dropout_mask(self, param_size):
        return torch.ones(size=param_size,device=self.device) * (1 - self.dropout)


    def forward(self, token, previous_state=None):
        #batch_size, token_length, _ = token.size()

        embedded = self.embedding(token)


        #mask = self.calculate_dropout_mask(layer_size=(batch_size, self.LSTM.hidden_size))
        for name, param in self.Var_LSTM.named_parameters():
            if 'weight_hh' in name:
                param.data *= self.calculate_dropout_mask(param.data.size())

        LSTM_output, hidden_layer = self.Var_LSTM(embedded, previous_state)

        # note that we should be taking the last layer of the lstm, but since we have only
        # a single layer, by default it's the last one and we don't need to "filter"
        output = self.output(LSTM_output).permute(0,2,1)

        return output, hidden_layer
            


class My_AvSGD(optim.SGD):
    def __init__(self, params, lr=4, validation_window=5, patience=5, threshold=0.01):
        super(My_AvSGD, self).__init__()

        self.validation_window = validation_window
        self.patience = patience
        self.threshold = threshold
        self.val_losses = []
        self.lr = lr

    def insert_loss(self, loss):
        self.val_losses.append(loss)

    
    def step(self):

        if len(self.val_losses) >= self.validation_window:
            average_loss = sum(self.val_losses) / self.validation_window

            # compares average loss to the minimum of a sliding window, also add an epsilon(threshold) to verify monotomy
            # if average is bigger than minimum loss achieved, slow down by lowering the lr
            if average_loss > min(self.val_losses[-self.validation_window - self.patience : -self.patience]) - self.threshold:
                self.lr *= 0.1
                ''' could edit the losses list to save space by doing this, but i'm not sure it's worth nor i'm sure it works, so i will keep it standard for now
                if(len(self.val_losses) > self.validation_window + self.patience):
                new_losses = self.val_losses[1 : self.validation_window + self.patience]'''

        # call SGD.step() as it contains the main points to update the model
        super(My_AvSGD, self).step()

        