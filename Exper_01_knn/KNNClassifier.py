# KNNClassifier.py
# hustccc
# 2020/4/14
import numpy as np
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import time
import operator


def KNNClassify(newInput, dataset, labels, k, weight, k_next):
    numSamples = dataset.shape[0]
    """step1: compute distance"""
    diff = np.tile(newInput,
                   (numSamples, 1)) - dataset  
    squaredist = diff**2
    distance = (squaredist.sum(axis=1))**0.5  
    """step2：rank base on distance"""
    sortedDistance = distance.argsort()

    classCount = {}
    temp=[]

    for i in range(k):
        votelabel = labels[sortedDistance[i]]
        temp = np.append(temp, votelabel)
        if weight == "uniform":
            classCount[votelabel] = classCount.get(votelabel, 0) + 1
        elif weight == "distance":
            classCount[votelabel] = classCount.get(
                votelabel, 0) + (1 / distance[sortedDistance[i]])
        else:
            print("Error！")
            print("\"uniform\"num_mux\"distance\"distance_min")
            break
    k_next.append(temp)
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    if weight == "uniform":
        print("%d new point:" % k, classCount)
        print("class:", sortedClassCount[0][0])

    elif weight == "distance":
        print("%d new point：" % k, classCount)
        print("class:", sortedClassCount[0][0])

    return sortedClassCount[0][0]


if __name__ == '__main__':
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    test_num = 1000  #num of test sample
    k_max = 10  # value of k
    result = np.zeros(test_num)  #save result
    rate = 0  # predict rate
    right = 0  # num of right predict
    predict = 0  # save predict
    report = []
    K_next = []
    """
    # predict begin
    t1 = time.time()
    for i in range(0, test_num):
        predict = KNNClassify(x_test[i],
                              x_train,
                              y_train,
                              7,
                              "distance",
                              k_next=K_next)
        result[i] = predict
        if (predict == y_test[i]):
            right += 1
        rate = right / (i + 1)
        print("index:%d :rate:%f" % (i, rate))
    t2 = time.time()
    print("final rate:%f" % (rate))
    print("time cost:%d" % (t2 - t1))

    print(result)
    print(K_next)
    """
    
    # predict begin:
    for j in range(1, k_max):
        t1 = time.time()
        right=0
        for i in range(0, test_num):
            predict = KNNClassify(x_test[i], x_train, y_train, j, "distance",k_next=K_next)
            result[i]=predict
            if (predict == y_test[i]):
                right += 1
            rate = right / (i + 1)
            print("K:%d->index:%d:rate:%f" % (j, i, rate))
        t2 = time.time()
        print("K:%d->final rate:%f" % (j, rate))
        print("time cost:%d" % (t2 - t1))
        report = np.append(report, (1-rate))
    print(report)

    x_label=np.arange(1,k_max)
    plt.title("Misclassification Rate")
    plt.xlabel("k")
    plt.ylabel("miss rate")
    plt.plot(x_label,report)
    plt.show()
