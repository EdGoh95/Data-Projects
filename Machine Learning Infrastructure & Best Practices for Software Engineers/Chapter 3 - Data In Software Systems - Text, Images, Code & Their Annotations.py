#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 3: Data in Software Systems - Text, Images, Code and Their Annotations
"""
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

#%% MNIST Dataset
MNIST_train = MNIST('./', train = True, download = True)
MNIST_test = MNIST('./', train = False, download = True)

plt.figure()
for i in range(9):
    MNIST_train_image, MNIST_train_label = MNIST_train[i]
    plt.subplot(330+1+i)
    plt.imshow(MNIST_train_image, cmap = plt.get_cmap('gray'))

#%% FashionMNIST Dataset
FashionMNIST_train = FashionMNIST('./', train = True, download = True)
FashionMNIST_test = FashionMNIST('./', train = False, download = True)

plt.figure()
for j in range(9):
    FashionMNIST_train_image, FashionMNIST_train_label = FashionMNIST_train[j]
    plt.subplot(330+1+j)
    plt.imshow(FashionMNIST_train_image, cmap = plt.get_cmap('gray'))

#%% CIFAR10 Dataset
CIFAR10_train = CIFAR10('./', train = True, download = True)
CIFAR10_test = CIFAR10('./', train = False, download = True)

plt.figure()
for k in range(9):
    CIFAR10_train_image, CIFAR10_train_label = CIFAR10_train[k]
    plt.subplot(330+1+k)
    plt.imshow(CIFAR10_train_image)