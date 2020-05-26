# -*- coding: utf-8 -*-
"""
Data procession for ML Lab 2
2020/5/23
Manjaro
hustccc
"""

import os, re, time
import threading, queue
import jieba
import random

lock = threading.Lock()
FLAG = False

#read stop word list
stop_word = open('./stop_word', encoding='utf-8').read().split('\n')

def procession(label, path):
    data_dir = './data'
    with open(path, 'rb') as f:
            text = f.read()
            text = text.decode('gbk', 'ignore')
    new_dir = '{}/{}'.format(data_dir, label)
    new_path = '{}/{}_{}.txt'.format(new_dir, 
                path.split('/')[-2], path.split('/')[-1])
    new_text = text.encode('utf-8', 'ignore').decode('utf-8')
    new_text = re.sub(r'[^\u4e00-\u9fa5]', '', new_text)
    new_text = jieba.cut(new_text, cut_all=False)
    new_text = ','.join([word for word in new_text if word not in stop_word])
    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(new_text)

class Worker(threading.Thread):
    def __init__(self, q):
        threading.Thread.__init__(self)
        self.q = q
    def run(self):
        line_index=0
        while 1:
            if not self.q.empty():
                lock.acquire()
                label, path = self.q.get()
                lock.release()
                procession(label, path)
                line_index+=1
                print("Procession [2] index: %d label: %s path: %s" %(line_index,label,path))
            else:
                time.sleep(1)
            '''
            if FLAG:
                exit()
                '''

def GenerateData():
    new_dir_1 = './data/SPAM'
    new_dir_2 = './data/HAM'
    if not os.path.exists(new_dir_1):
        os.makedirs(new_dir_1)
        os.makedirs(new_dir_2)
    else:
        print('Data already exists')
        return
    index_path = '../trec06c/full/index'
    with open(index_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    # lines = lines[:int(0.7*len(lines))]
    data = list()
    line_index=0
    for line in lines:
        label = line.split(' ')[0]
        path = '../trec06c'+line.split(' ')[1].replace('\n', '')[2:]
        data.append([label.upper(), path])
        line_index+=1
        print("Read line [0] index: %d label: %s" %(line_index,label.upper()))
    Q = queue.Queue()
    line_index=0
    for label, path in data:
        Q.put([label, path])
        line_index+=1
        print("In queue [1] index: %d label: %s" %(line_index,label))
    ws = list()
    for j in range(100):
        w = Worker(Q)
        w.start()
        ws.append(w)
    while not Q.empty():
        pass
    #print('Queue empty!')
    '''
    for w in ws:
        w.join()
    '''
    print('Done!!!')
    exit()
    '''
    global FLAG
    FLAG = True
    '''
    
if __name__=="__main__":
    #start generate data
    GenerateData()
