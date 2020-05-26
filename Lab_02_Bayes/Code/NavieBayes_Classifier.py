# -*- coding: utf-8 -*-
"""
NavieBayes Classifier for ML Lab 2
2020/5/23
Manjaro
hustccc
"""
import os, time, random
import numpy as np

s = time.time()

def LoadData(rate=1):
    DataDir = './data'
    labels = os.listdir(DataDir)
    data = list()
    print("Start load data...")
    for label in labels:
        label_dir = DataDir+os.sep+label
        filenames = os.listdir(label_dir)
        index=0
        for filename in filenames:
            data_path = label_dir + os.sep + filename
            with open(data_path, encoding='utf-8') as f:
                text = f.read()
            data.append([1 if label == 'SPAM' else 0, text.split(',')])
            index+=1
            print("Load data [%d] filename: %s"%(index,filename))
    random.shuffle(data)
    print('Read data done，time cost：{}'.format(time.time()-s))
    return data[:int(len(data)*rate)]

def BuildWordset(data):
    wordset = set()
    for label, text in data:
        wordset |= set(text)
    return list(wordset)

def MakeWordVector(wordset, data):
    wordict = {word:k for k, word in enumerate(wordset)}
    print('length of word list：{}'.format(len(wordset)))
    #word_vector = np.zeros([len(data), len(wordset)])
    #word_vector = np.zeros([len(data), len(wordset)],dtype='float32')
    word_vector = np.zeros([len(data), len(wordset)],dtype='int8')
    labels = np.zeros(len(data))
    print("Start make word vector...")
    index=0
    for k, [label, text] in enumerate(data):
        for word in text:
            if word in wordict:
                word_vector[k, wordict[word]] = 1
        labels[k] = label
        index+=1
        print("Make word vector [%d] label: %s"%(index,label))
    print('make vertor done, time cost：{}'.format(time.time()-s))
    return word_vector, labels

def TrainModule(data, labels):
    print('Training\n=====>')
    Pspam = sum(labels) / len(labels)
    Pham = 1 - Pspam
    SN = np.ones(data.shape[1])
    HN = np.ones(data.shape[1])
    index=0
    for k, d in enumerate(data):
        if labels[k]:
            SN += d
            index+=1
            print("Train index [%d] label [%d]"%(index,labels[k]))
        else:
            HN += d
            index+=1
            print("Train index [%d] label [%d]"%(index,labels[k]))
    PS = SN / sum(SN)
    PH = HN / sum(HN)
    print('Train done, time cost：{}'.format(time.time()-s))
    return Pspam, Pham, PS, PH

def PredictModule(data, Pspam, Pham, PS, PH):
    print('Testing\n=====>')
    PS = np.log(PS)
    PH = np.log(PH)
    Pspam = np.math.log(Pspam)
    Pham = np.math.log(Pham)
    '''
    predict_result = [1 if (Pspam+sum(d*PS)) >= (Pham+sum(d*PH)) else 0 
                   for d in data]
                   '''
    predict_result = []
    index=0
    for d in data:
        if (Pspam+sum(d*PS)) >= (Pham+sum(d*PH)):
            predict_result.append(1)
            index+=1
            print("Predict [%d] %s" %(index,"spam"))
            pass
        else:
            predict_result.append(0)
            index+=1
            print("Predict [%d] %s" %(index,"hpam"))
            pass
            
    return predict_result


if __name__=="__main__":
    train_rate=0.7
    data = LoadData()
    wordset = BuildWordset(data)
    word_vector, labels = MakeWordVector(wordset, data)
    x_train = word_vector[:int(train_rate*word_vector.shape[0]), :]
    y_train = labels[:int(train_rate*labels.shape[0])]
    x_test = word_vector[int(train_rate*word_vector.shape[0]):, :]
    y_test = labels[int(train_rate*labels.shape[0]):]
    print("Train data shape:")
    print(word_vector.shape[0])
    print("Train label shape:")
    print(labels.shape[0])
    #Pspam, Pham, PS, PH = TrainModule(x_train, labels)
    Pspam, Pham, PS, PH = TrainModule(x_train, y_train)
    #predict = PredictModule(word_vector[int(train_rate*word_vector.shape[0]):, :], Pspam, Pham, PS, PH)
    #accuary = np.mean(predict==labels[int(train_rate*labels.shape[0]):])
    predict = PredictModule(x_test, Pspam, Pham, PS, PH)
    accuary = np.mean(predict==y_test)
    print("Test done")
    print('Correct Rate：{}'.format(accuary))
    print('Total time cost：{}'.format(time.time()-s))
