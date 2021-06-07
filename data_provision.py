import numpy as np
import struct
# import matplotlib.pyplot as plt
import os


def load_mnist_images(file_name):
    ##   在读取或写入一个文件之前，你必须使用 Python 内置open()函数来打开它。##
    ##   file object = open(file_name [, access_mode][, buffering])          ##
    ##   file_name是包含您要访问的文件名的字符串值。                         ##
    ##   access_mode指定该文件已被打开，即读，写，追加等方式。               ##
    ##   0表示不使用缓冲，1表示在访问一个文件时进行缓冲。                    ##
    ##   这里rb表示只能以二进制读取的方式打开一个文件                        ##
    binfile = open(file_name, 'rb') 
    ##   从一个打开的文件读取数据
    buffers = binfile.read()
    ##   读取image文件前4个整型数字
    magic, num, rows, cols = struct.unpack_from('>IIII',buffers, 0)
    ##   整个images数据大小为60000*28*28
    bits = num * rows * cols
    ##   读取images数据
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    ##   关闭文件
    binfile.close()
    ##   转换为[60000,784]型数组
    images = np.reshape(images, [num, rows * cols])
    return images


def load_mnist_labels(file_name):
    ##   打开文件
    binfile = open(file_name, 'rb')
    ##   从一个打开的文件读取数据    
    buffers = binfile.read()
    ##   读取label文件前2个整形数字，label的长度为num
    magic, num = struct.unpack_from('>II', buffers, 0) 
    ##   读取labels数据
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    ##   关闭文件
    binfile.close()
    ##   转换为一维数组
    labels = np.reshape(labels, [num])
    return labels 


def load_mnist_data(file_path='./dataset/MNIST/', if_sort=False):
    filename_train_images = os.path.join(file_path, 'raw/train-images-idx3-ubyte')
    filename_train_labels = os.path.join(file_path, 'raw/train-labels-idx1-ubyte') 
    filename_test_images = os.path.join(file_path, 'raw/t10k-images-idx3-ubyte')
    filename_test_labels = os.path.join(file_path, 'raw/t10k-labels-idx1-ubyte')
    train_images=load_mnist_images(filename_train_images)
    train_labels=load_mnist_labels(filename_train_labels)
    test_images=load_mnist_images(filename_test_images)
    test_labels=load_mnist_labels(filename_test_labels)

    sorted_indices = np.argsort(train_labels)
    sorted_train_images = train_images[sorted_indices]
    sorted_train_labels = train_labels[sorted_indices]

    sorted_indices = np.argsort(test_labels)
    sorted_test_images = test_images[sorted_indices]
    sorted_test_labels = test_labels[sorted_indices]

    return sorted_train_images, sorted_train_labels, sorted_test_images, sorted_test_labels



def get_split_data(dataset_dir='./dataset/MNIST/', split_num=100):
    filename_train_images = './dataset/MNIST/raw/train-images-idx3-ubyte'
    filename_train_labels = './dataset/MNIST/raw/train-labels-idx1-ubyte'
    filename_test_images = './dataset/MNIST/raw/t10k-images-idx3-ubyte'
    filename_test_labels = './dataset/MNIST/raw/t10k-labels-idx1-ubyte'
    
    split_train_images, split_train_labels, split_test_images, split_test_labels = [], [], [], []
    


if __name__ == '__main__':
    filename_train_images = './dataset/MNIST/raw/train-images-idx3-ubyte'
    filename_train_labels = './dataset/MNIST/raw/train-labels-idx1-ubyte'
    filename_test_images = './dataset/MNIST/raw/t10k-images-idx3-ubyte'
    filename_test_labels = './dataset/MNIST/raw/t10k-labels-idx1-ubyte'
    train_images=load_mnist_images(filename_train_images)
    train_labels=load_mnist_labels(filename_train_labels)
    test_images=load_mnist_images(filename_test_images)
    test_labels=load_mnist_labels(filename_test_labels)

    # fig, ax = plt.subplots(
    # nrows=2,
    # ncols=5,
    # sharex=True,
    # sharey=True, )
 
    # ax = ax.flatten()
    # for i in range(10):
    #     img = train_images[train_labels == i][0].reshape(28, 28)
    #     ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()