import torch
import torch.nn as nn
import TextCNN
import dataset
import train
import LSTM
import Fasttext
import BiLSTM_Attention
import matplotlib.pyplot as plt
import time
import os
from datetime import timedelta
import warnings
import argparse
import os
warnings.filterwarnings("ignore")

def using_time(start):
    end_time = time.time()
    time_dif = end_time - start
    return timedelta(seconds=int(round(time_dif)))


def get_model(model, Title):
    embedding_dim = len(Title.vocab.vectors[0])
    embedding_weight = Title.vocab.vectors

    if model == 'TextCNN':
        model = TextCNN.TextCNN(max_length=20,
                                vocab_size=len(Title.vocab),
                                embedding_dim=embedding_dim,
                                embedding_weight=embedding_weight,
                                output_size=10)
    elif model == 'LSTM':
        model = LSTM.LSTM(vocab_size=len(Title.vocab),
                          embedding_dim=embedding_dim,
                          embedding_weight=embedding_weight,
                          hidden_size=100,
                          num_layers=4,
                          output_size=10)
    elif model == 'BiLSTM_Attention':
        model = BiLSTM_Attention.BiLSTM_Attention(vocab_size=len(Title.vocab),
                                 embedding_dim=embedding_dim,
                                 embedding_weight=embedding_weight,
                                 hidden_size=100,
                                 num_layers=4,
                                 output_size=10)
    else:
        model = Fasttext.fasttext(vocab_size=len(Title.vocab),
                                embedding_dim=embedding_dim,
                                embedding_weight=embedding_weight,
                                hidden_size=100,
                                output_size=10)
    return model


def run(model_name, model, train_iter, test_iter, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    StepLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    acc, loss = train.test(model, train_iter, criterion)
    train_acc.append(acc)
    train_loss.append(loss)
    acc, loss = train.test(model, test_iter, criterion)
    test_acc.append(acc)
    test_loss.append(loss)
    print('train_acc: %.4f' %(train_acc[-1]), 'test_acc: %.4f'%(test_acc[-1]))

    if not os.path.exists('./model'):
        os.mkdir('./model')


    start = time.time()
    for epoch in range(epochs):
        acc, loss = train.train(model, train_iter, criterion, optimizer)
        train_acc.append(acc)
        train_loss.append(loss)

        acc, loss = train.test(model, test_iter, criterion)

        if acc >= max(test_acc):
            torch.save(model, './model/' + model_name + '.pth')

        test_acc.append(acc)
        test_loss.append(loss)

        print('Epoch %d:' %(epoch+1), ' train_acc: %4f' %(train_acc[-1]),  '  test_acc: %4f' %(test_acc[-1]),
              ' train_loss: %.4f' %(train_loss[-1]), '  test_loss: %.4f' %(test_loss[-1]),
              '  learning rate: %f' %(optimizer.param_groups[0]['lr']))
        StepLR.step(test_loss[-1])


    print("using time: ", using_time(start)/epochs, "/epoch")
    print("best test acc:", max(test_acc))

    train_acc.pop(0)
    test_acc.pop(0)
    train_loss.pop(0)
    test_loss.pop(0)

    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.plot(train_loss)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.xlim(xmin=0, xmax=epochs+1)
    plt.title('The accuracy of ' +model_name+ ' model')
    plt.legend(["train acc", "test acc", "train loss"])
    plt.show()

def main(config):
    os.system('cls')
    print('Loading dataset')
    start = time.time()
    train_iter, test_iter, Title, Label = dataset.get_dataset()
    print('Done! using time:', using_time(start))

    name = config.model
    times = 20 if not config.times else config.times
    model = get_model(name, Title)
    run(name, model, train_iter, test_iter, times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input the model name and train times')
    parser.add_argument('model', choices=['TextCNN', 'LSTM', 'fasttext', 'BiLSTM_Attention'], help='input the model name')
    parser.add_argument('-t', '--times', type=int, help='input the train times')
    args = parser.parse_args()
    
    main(args)





