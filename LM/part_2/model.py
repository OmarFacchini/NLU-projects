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
    def __init__(self, vocab_size, padding_index, train_criterion, eval_criterion, embedding_dim=300, hidden_dim=200, regularization=2, device='cuda'):
        super(Variational_Dropout_LM_LSTM, self).__init__()

        self.hidden_layers_size = hidden_dim
        self.embedded_layer_size = embedding_dim
        self.output_size = vocab_size
        self.padding_index = padding_index
        self.number_of_layers = 1
        self.dropout = 0.2
        self.regularization = regularization
        self.device = device

        self.dropout_mask1 = None
        self.dropout_mask2 = None
        
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


    def calculate_dropout_mask1(self, x):
        # create tensor full of ones of size equal to x(basically just used to ensure dimensionality)
        temp_mask = torch.ones_like(x)

        # fill(edit) the values of the mask to be 1-dropout
        temp_mask = temp_mask.fill_(1 - self.dropout)

        # use a bernoulli distribution to generate binary mask, since all values are 1-dropout
        # they all have the same probability to be sampled, therefore this is basically a uniform distribution
        # but bernoulli allows direct implementation of varying probabilities so i opted on that rather than the uniform
        self.dropout_mask1 = torch.bernoulli(temp_mask)


    def calculate_dropout_mask2(self, x):
        # create tensor full of ones of size equal to x(basically just used to ensure dimensionality)
        temp_mask = torch.ones_like(x)

        # fill(edit) the values of the mask to be 1-dropout
        temp_mask = temp_mask.fill_(1 - self.dropout)

        # use a bernoulli distribution to generate binary mask, since all values are 1-dropout
        # they all have the same probability to be sampled, therefore this is basically a uniform distribution
        # but bernoulli allows direct implementation of varying probabilities so i opted on that rather than the uniform
        self.dropout_mask2 = torch.bernoulli(temp_mask)

        #could also use the one implemented in the paper, should be the same:
        #self.dropout_mask2 = x.weight.data.new().resize_((x.weight.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(x.weight) / (1 - self.dropout)


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.number_of_layers, batch_size, self.hidden_layers_size).zero_().to(self.device),
                  weight.new(self.number_of_layers, batch_size, self.hidden_layers_size).zero_().to(self.device))
     
        return hidden


    def apply_variational_dropout(self, x, mask, dropout):
        # note: x*mask multiplies my tensor with the mask, actually applying the dropout
        # we then divide by 1-self.dropout to maintain magnitude of the values we keep
        return (x * mask) / (1 - dropout)


    def forward(self, token, previous_state=None):
        #batch_size, token_length, _ = token.size()

        embedded = self.embedding(token)

        # check if dropout mask is calculated, else, calculate it
        # note: at first loop, we enter this if as it is initialized to None
        # subsequently, since we always want to perform a full forward + backward pass before computing again the mask
        # we set it to None after the backward, and calculate it when restarting the forward

        # note2: i cannot initialize when initializing the class as i need the shape of the tensor inputted to the dropout
        # and i either add 2 tensors to the class just for that, or i do this if, i chose the latter

        # note3: this should not create issues with in-batch forward as once we calculate it, it's not None anymore
        if self.dropout_mask1 is None:
            self.calculate_dropout_mask1(embedded)

        if self.training:
            embedded = self.apply_variational_dropout(embedded, self.dropout_mask1, self.dropout)

        LSTM_output, hidden_layer = self.Var_LSTM(embedded, previous_state)

        # check if dropout mask is calculated, else, calculate it
        # note: at first loop, we enter this if as it is initialized to None
        # subsequently, since we always want to perform a full forward + backward pass before computing again the mask
        # we set it to None after the backward, and calculate it when restarting the forward

        # note2: i cannot initialize when initializing the class as i need the shape of the tensor inputted to the dropout
        # and i either add 2 tensors to the class just for that, or i do this if, i chose the latter

        # note3: this should not create issues with in-batch forward as once we calculate it, it's not None anymore 
        if self.dropout_mask2 is None:
            # here we could pass LSTM_output.weight_hh_l0 meaning weights of hidden-hidden(hh) connections of first layer(l0)
            # while what we did down here is pass all the weights and biases of the layer
            self.calculate_dropout_mask2(LSTM_output)

        if self.training:
            LSTM_output = self.apply_variational_dropout(LSTM_output, self.dropout_mask2, self.dropout)

        # note that we should be taking the last layer of the lstm, but since we have only
        # a single layer, by default it's the last one and we don't need to "filter"
        output = self.output(LSTM_output).permute(0,2,1)

        return output, hidden_layer 

  