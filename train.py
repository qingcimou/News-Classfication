import torch
from sklearn import metrics
import numpy as np

def train(model, train_iter, criterion, optimizer):
    running_loss = []
    correct = 0
    total = 0
    model.train()
    for batch_idx, data in enumerate(train_iter):
        inputs, label = data.Title, data.Label
        label = label.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        '''
        l1_regularization = torch.tensor([0],dtype =torch.float32, device='cuda')
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1)
        loss = loss+0.0000001*l1_regularization
        '''

        loss.backward()
        optimizer.step()

        _, predict = torch.max(outputs, dim=1)
        total += label.size(0)
        correct += (predict == label).sum().item()
        running_loss.append(loss.item())
    acc = correct / total

    running_loss = np.array(running_loss).mean()
    return acc, running_loss


def test(model, test_iter, criterion, test=False):
    running_loss = []
    correct = 0
    total = 0
    labels = np.array([], dtype=int)
    predicts = np.array([], dtype=int)
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_iter):
            inputs, label = data.Title, data.Label
            label = label.long()
            outputs = model(inputs)
            loss = criterion(outputs, label)


            _, predict = torch.max(outputs, dim=1)

            total += label.size(0)
            correct += (predict == label).sum().item()
            running_loss.append(loss.item())

            label = label.data.cpu().numpy()
            labels = np.append(labels, label)

            predict = predict.cpu().numpy()
            predicts = np.append(predicts, predict)

        acc = correct / total
        running_loss = np.array(running_loss).mean()
    if test:
        report = metrics.classification_report(labels, predicts)
        confusion = metrics.confusion_matrix(labels, predicts)
        return acc, report, correct
    return acc, running_loss
