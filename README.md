# News_Classfication

## 介绍
毕业设计，基于卷积神经网络的新闻分类，包含TextCNN、LSTM、BiLSTM+Attention、fasttext

P.S:代码写的比较乱，数据集与词向量均来自互联网，有侵权问题可以联系我

## 基本环境
|名称|版本|
|----|----|
|Python|3.6.6|
|Pytorch|1.6.0|
|torchtext|0.6.0|
|Sklearn|0.20.3|
|Seaborn|0.11.0|


## 使用
```
# 训练
# python run.py [-h] [-t TIMES] {TextCNN,LSTM,fasttext,BiLSTM_Attention}
# 示例
# python run.py TextCNN -t 10
# python run.py LSTM

# 评价(先训练一次得到模型)
# python evaluate.py [-h] {TextCNN,LSTM,fasttext,BiLSTM_Attention}
# 示例
# python evaluate.py LSTM
```



