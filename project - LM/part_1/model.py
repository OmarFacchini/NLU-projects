import torch
import torch.nn as nn
import math
    
class LM_LSTM(nn.Module):

    def __init__(self, vocab_size, padding_index, train_criterion, eval_criterion, embedding_dim=300, hidden_dim=200):
        super(LM_LSTM, self).__init__()

        self.hidden_layers_size = embedding_dim
        self.embedded_layer_size = hidden_dim
        self.output_size = vocab_size
        self.padding_index = padding_index
        self.number_of_layers = 1
        
        self.criterion_train = train_criterion
        self.criterion_eval  = eval_criterion

        # simple lookup table that stores embeddings of a fixed dictionary and size
        self.embedding = nn.Embedding(num_embeddings=self.output_size, 
                                      embedding_dim=self.embedded_layer_size, 
                                      padding_idx=self.padding_index)

        # drop some random values with probability p
        self.dropout = nn.Dropout(p=0.2)

        # LSTM: apply memory RNN to an input
        # note: could adde the parameter droput, but it applies to all LSTM layers EXCEPT the last one, so I would rather have it directly outside and manipulate however I want
        # for clarity
        self.LSTM = nn.LSTM(input_size=self.embedded_layer_size,
                            hidden_size=self.hidden_layers_size,
                            num_layers=self.number_of_layers, 
                            bidirectional=True)
        
        self.dropout2 = nn.Dropout(p=0.2)

        # linear layer to map back to the uoutput space
        self.output = nn.Linear(self.hidden_layers_size, self.output_size)


    def init_weights(self, mat):
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
    

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(1, batch_size, self.lstm.hidden_size).zero_())
        return hidden


    def forward(self, token, previous_state):
        embedded = self.embedding(token)
        embedded = self.dropout(embedded)

        LSTM_output, hidden_layer = self.LSTM(embedded, previous_state)
        LSTM_output = self.dropout2(LSTM_output)

        output = self.output(LSTM_output).permute(0,2,1)

        return output, hidden_layer
    

    def train(self, data, optimizer, clip=5):
        self.train()
        loss = 0
        total_loss = 0
        number_of_tokens = []

        for sample in data:
            hidden = self.init_hidden(sample['source'].size(0))
            optimizer.zero_grad() # Zeroing the gradient
            output, hidden = self(sample['source'], hidden)
            loss = self.criterion_train(output, sample['target'])
            total_loss += loss.item() * sample["number_tokens"]
            
            number_of_tokens.append(sample["number_tokens"])
            loss.backward() # Compute the gradient
            # clip the gradient to avoid explosioning gradients
            nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step() # Update the weights

        return total_loss/sum(number_of_tokens)
    


    def validation(self, data):
        self.eval()

        with torch.no_grad():
            total_loss = 0
            number_of_tokens = []
            for sample in data:
                hidden = self.init_hidden(sample['source'].size(0))
                output, hidden = self(sample['source'], hidden)

                #could remove loss and directly edit the total_loss but this looks cleaner and clearer
                loss = self.criterion_eval(output, sample['target'])
                total_loss += loss.item() * sample["number_tokens"]
                
                number_of_tokens.append(sample["number_tokens"])   

        # calculate perplexity and averaged loss that will be returned as measures for performance
        perplexity = math.exp(total_loss / sum(number_of_tokens))
        average_loss = total_loss/sum(number_of_tokens)

        return perplexity, average_loss

        