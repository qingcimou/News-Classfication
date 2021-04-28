import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self,
                 max_length,
                 vocab_size,
                 embedding_dim,
                 embedding_weight,
                 output_size,
                 filtter_num=16,
                 kernellist=(2, 3, 4),
                 dropout=0.3):
        super(TextCNN, self).__init__()

        self.emb = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weight !=  None:
            self.emb.weight.data.copy_(embedding_weight)
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(1, filtter_num, (kernel, embedding_dim)),
                                                 nn.ReLU(),
                                                 nn.MaxPool2d((max_length-kernel+1, 1))
                                                 )
                                    for kernel in kernellist
                                   ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filtter_num*len(kernellist), output_size)
    def forward(self, x):
        x = self.emb(x)
        #x = self.dropout(x)

        x = x.unsqueeze(1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)