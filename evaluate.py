import torch
import dataset
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import os
import argparse

#define
target_names = ['财经', '房产', '股票', '教育', '科技', '社会', '时政', '体育', '游戏', '娱乐']
sns.set(font='SimHei', font_scale=1.0)
#['finance', 'realty', 'stocks', 'education', 'science', 'society', 'politics', 'sports', 'game', 'entertainment']

def using_time(start):
    end_time = time.time()
    time_dif = end_time - start
    return timedelta(seconds=int(round(time_dif)))

def predict(model, test_iter):
    model.eval()
    predicts, labels = np.array([], dtype=int), np.array([], dtype=int)
    for batch_idx, data in enumerate(test_iter):
        inputs, label = data.Title, data.Label

        label = label.type(torch.int32)
        label = label.cpu().numpy()
        labels = np.append(labels, label)


        outputs = model(inputs)
        _, predict = torch.max(outputs, dim=1)
        predict = predict.cpu().numpy()
        predicts = np.append(predicts, predict)

    return predicts, labels


def evalute(name, pred, label):
    print('-------------------------------'+name+'-------------------------------')
    print(classification_report(label, pred, target_names=target_names))
    matrix = confusion_matrix(label, pred)
    ax = sns.heatmap(matrix, xticklabels=target_names, yticklabels=target_names, fmt='.20g', cmap='Greys_r',annot=True)
    ax.set_title(name+' model')
    plt.show()


def main(config):
    os.system('cls')
    print('Loading dataset')
    start = time.time()
    train_iter, test_iter, Title, Label = dataset.get_dataset()
    print('Done! using time:', using_time(start))
    path = './model/'

    name = config.model

    model = torch.load(path + name + '.pth')
    predicts, labels = predict(model, test_iter)
    evalute(name, predicts, labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input the model name')
    parser.add_argument('model', choices=['TextCNN', 'LSTM', 'fasttext', 'BiLSTM_Attention'], help='input the model name')
    args = parser.parse_args()
    main(args)