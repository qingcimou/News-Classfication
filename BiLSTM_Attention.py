import torch
import torch.nn as nn

class BiLSTM_Attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding_weight,
                 hidden_size,
                 num_layers,
                 output_size):
        super(BiLSTM_Attention, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=embedding_dim)
        if embedding_weight != None:
            self.emb.weight.data.copy_(embedding_weight)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.w = nn.Parameter(torch.zeros(hidden_size*2))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def attention(self, out):
        #out = torch.Tensor.reshape(out, )
        M = self.tanh(out)
        alpha = self.softmax(torch.matmul(M, self.w)).unsqueeze(-1)
        out = out*alpha
        return torch.sum(out, 1)


    def forward(self, x):
        x = self.emb(x)
        out, (h_n, c_n) = self.lstm(x)
        out = self.relu(self.attention(out))
        out = self.dropout(out)
        return self.fc(out)


