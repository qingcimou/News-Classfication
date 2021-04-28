import torch
import torchtext
import random
from torchtext import data
import jieba
import pkuseg

#define
MAX_LINGTH = 20
BATCH_SIZE = 1280
seg = pkuseg.pkuseg(model_name='web')
with open('./datasets/hit_stopwords.txt', encoding='utf-8') as f:
    stopwords = f.read().split('\n')

def tokenizer(text):
    return [token for token in seg.cut(text) if token not in stopwords]

def get_dataset():
    Title = data.Field(sequential=True,
                       use_vocab=True,
                       tokenize=tokenizer,
                       fix_length=MAX_LINGTH,
                       lower=True,
                       batch_first=True
                       )

    Label = data.LabelField(dtype=torch.float)



    train, test = data.TabularDataset.splits(path='./datasets/',
                                            train='train.csv',
                                            test='test.csv',
                                            format='csv',
                                            fields=[('Title', Title),
                                                    ('Label', Label)],
                                            skip_header=True
                                            )

    vectors_path = torchtext.vocab.Vectors(name='45000-small.txt',
                                           cache='./Word2Vec')

    Title.build_vocab(train,
                      max_size=50000,
                      min_freq=3,
                      vectors=vectors_path
                      )
    Label.build_vocab(train)


    train_iter = data.BucketIterator(dataset=train,
                                     batch_size=BATCH_SIZE,
                                     sort=False,
                                     shuffle=True,
                                     device=torch.device("cuda")
                                     )

    test_iter = data.BucketIterator(dataset=test,
                                     batch_size=BATCH_SIZE,
                                     sort=False,
                                     shuffle=True,
                                     device=torch.device("cuda")
                                     )





    return train_iter, test_iter, Title, Label

'''
if __name__ == "__main__":
    train_iter, val_iter, test_iter, Title, Label = get_dataset()
    print(len(Title.vocab.stoi))
    print(len(Title.vocab.vectors[0]))
    print(Label.vocab.stoi)
'''


