import torch
from torch import nn

class fasttext(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding_weight,
                 hidden_size,
                 output_size):
        super(fasttext, self).__init__()

        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim = embedding_dim)
        if embedding_weight != None:
            self.emb.weight.data.copy_(embedding_weight)

        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.mean(1)
        h = self.relu(self.fc1(x))

        out = self.fc2(h)

        return out
