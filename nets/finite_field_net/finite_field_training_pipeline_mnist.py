#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.path.append('../../')

import numpy as np
from datasets import load_all_data_mnist, load_all_data_cifar10, load_all_data_fashion_mnist,\
    load_all_data_apply_vgg_cifar10
from utils import create_batch_data
import modules
import layers
from criterions import FiniteFieldMSELoss


# In[2]:


BATCH_SIZE = 256
EPOCH = 1
PRINT = 10
FLATTEN = False
# 0, MNIST; 1, FashionMNIST; 2, CIFAR10; 3, VGG-CIFAR10
DATASET_MODE = 0

QUANTIZATION_INPUT = 8
QUANTIZATION_WEIGHT = 16
QUANTIZATION_BATCH_SIZE = 8
LR = 7
PRIME = 684502462494449

# In[3]:


# data fetching
load_path = '../../data'
if DATASET_MODE == 0:
    train_data, train_label, test_data, test_label = load_all_data_mnist(load_path, QUANTIZATION_INPUT,
                                                                         QUANTIZATION_WEIGHT, PRIME, flatten=FLATTEN)
elif DATASET_MODE == 1:
    train_data, train_label, test_data, test_label = load_all_data_fashion_mnist(load_path, QUANTIZATION_INPUT,
                                                                                 QUANTIZATION_WEIGHT, PRIME,
                                                                                 flatten=FLATTEN)
elif DATASET_MODE == 2:
    train_data, train_label, test_data, test_label = load_all_data_cifar10(load_path, QUANTIZATION_INPUT,
                                                                           QUANTIZATION_WEIGHT, PRIME,
                                                                           flatten=FLATTEN)
elif DATASET_MODE == 3:
    train_data, train_label, test_data, test_label = load_all_data_apply_vgg_cifar10(load_path, QUANTIZATION_INPUT,
                                                                                     QUANTIZATION_WEIGHT,
                                                                                     PRIME,
                                                                                     flatten=FLATTEN)
else:
    train_data, train_label, test_data, test_label = None, None, None, None
train_data, train_label, test_data, test_label = create_batch_data(train_data, train_label, test_data, test_label,
                                                                   BATCH_SIZE)


# In[4]:


model_arr = [
    layers.FiniteFieldPiNetSecondOrderConvLayer(1, 6, (5, 5), QUANTIZATION_WEIGHT, PRIME, first_layer=True,
                                                quantization_bit_input=QUANTIZATION_INPUT),
    layers.FiniteFieldPiNetSecondOrderConvLayer(6, 6, (5, 5), QUANTIZATION_WEIGHT, PRIME),
    modules.Flatten(),
    layers.FiniteFieldPiNetSecondOrderLinearLayer(2400, 128, QUANTIZATION_WEIGHT, PRIME),
    layers.FiniteFieldLinearLayer(128, 10, QUANTIZATION_WEIGHT, PRIME)
]

model = modules.Network(model_arr)
criterion = FiniteFieldMSELoss(PRIME, QUANTIZATION_WEIGHT, QUANTIZATION_BATCH_SIZE)


# In[ ]:


for epoch in range(EPOCH):
    tot_loss = 0
    for train_idx, (train_data_batch, train_label_batch) in enumerate(zip(train_data, train_label)):
        # train
        preds = model.forward(train_data_batch)

        loss = criterion.forward(preds, train_label_batch)
        tot_loss += loss
        propagated_error = criterion.error_derivative()

        model.backprop(propagated_error)
        model.optimize(LR)
        print('epoch: {}, idx: {}, curr loss: {}'.format(epoch + 1, train_idx + 1, loss))
        if train_idx == 0 or (train_idx + 1) % PRINT == 0:
            if train_idx != 0:
                tot_loss = tot_loss / PRINT
            print('epoch: {}, idx: {}, avg loss: {}'.format(epoch + 1, train_idx + 1, tot_loss))
            tot_loss = 0

    tot_acc = 0
    tot_sample = 0
    for train_acc_idx, (test_data_batch, test_label_batch) in enumerate(zip(test_data, test_label)):
        # train accuracy
        preds = model.forward(test_data_batch)
        pred_args = np.argmax(preds, axis=1)

        tot_acc += np.count_nonzero(pred_args == test_label_batch)
        tot_sample += test_data_batch.shape[0]
    accuracy = tot_acc / tot_sample
    print('epoch: {}, accuracy: {}'.format(epoch + 1, accuracy))
