# make_dataset.py
# hustccc
# 2020/4/14
import numpy as np
import struct
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import time
import operator

# 训练集文件
train_images_idx3_ubyte_file = '/home/hustccc/Machine_Learning_Experiment/Exper_01_knn/train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '/home/hustccc/Machine_Learning_Experiment/Exper_01_knn/train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '/home/hustccc/Machine_Learning_Experiment/Exper_01_knn/t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '/home/hustccc/Machine_Learning_Experiment/Exper_01_knn/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(
        fmt_header, bin_data, offset)
    print('mirror:%d, num_of_image: %d, size: %d*%d' %
          (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(
        fmt_header
    )  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(
        image_size
    ) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image, offset, struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('done: %d' % (i + 1))
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data,
                                                offset)).reshape(
                                                    (num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)


#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
#plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('mirror:%d, num_of_image: %d' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('done: %d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    # 查看前十个数据及其标签以读取是否正确
    # for i in range(10):
    # print(train_labels[i])
    # plt.imshow(train_images[i], cmap='gray')
    # plt.pause(0.000001)
    # plt.show()
    # print('done')

    # reshape as (,784)
    x_train = np.reshape(train_images, (60000, 784))
    x_test = np.reshape(test_images, (10000, 784))
    y_train = train_labels
    y_test = test_labels
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # two values for x_train
    for i in range(0, 60000):
        for j in range(0, 784):
            if (x_train[i][j] > 127):
                x_train[i][j] = 1
            else:
                x_train[i][j] = 0

    # two values for x_test
    for i in range(0, 10000):
        for j in range(0, 784):
            if (x_test[i][j] > 127):
                x_test[i][j] = 1
            else:
                x_test[i][j] = 0
    
    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('x_test',x_test)
    np.save('y_test',y_test)

    print("done")