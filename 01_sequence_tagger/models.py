import torch.nn as nn
import torch.nn.functional as F

#===========================================================================
class BiLSTMTagger(nn.Module):
    def __init__(self, hidden_size, n_layers, n_classes, embedding_weights):
        super(BiLSTMTagger, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embedding_weights)
        self.bilstm = nn.LSTM(input_size=embedding_weights.shape[1],
                             hidden_size=hidden_size,
                             num_layers=n_layers,
                             bidirectional=True)
        self.linear = nn.Linear(in_features=2*hidden_size,
                               out_features=n_classes)
        
    def forward(self, seq_sentence):      
        embed_out = self.embedding_layer(seq_sentence)
        # embed_out: [len(seq_sentence), embed_dim]   
        bilstm_out, _ = self.bilstm(embed_out.view(len(seq_sentence), 1, -1))
        # out: [len(seq_sentence), 1, 2*hidden_size] 
        linear_output = self.linear(bilstm_out.view(len(seq_sentence), -1))
        # linear_output: [2*hidden_size, num_classes]        
        softmax_out = F.log_softmax(linear_output, dim=1)
        # softmax_out: [2*hidden_size, num_classes]       
        return softmax_out