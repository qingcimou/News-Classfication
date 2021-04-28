import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim,
                 embedding_weight,
                 hidden_size,
                 num_layers,
                 output_size):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim)
        if embedding_weight !=  None:
            self.emb.weight.data.copy_(embedding_weight)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.emb(x)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :].view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x)

