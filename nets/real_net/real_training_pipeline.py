#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from datasets import load_all_data_mnist, load_all_data_cifar10, load_all_data_fashion_mnist, \
    load_all_data_apply_vgg_cifar10
from utils import create_batch_data
import modules
import layers
from criterions import RealMSELoss


# In[7]:


BATCH_SIZE = 256
EPOCH = 5
LR = 0.01
PRINT = 10
FLATTEN = False
# 0, MNIST; 1, FashionMNIST; 2, CIFAR10; 3, VGG-CIFAR10
DATASET_MODE = 2


# In[8]:


# data fetching
load_path = '../../data'
if DATASET_MODE == 0:
    train_data, train_label, test_data, test_label = load_all_data_mnist(load_path, flatten=FLATTEN)
elif DATASET_MODE == 1:
    train_data, train_label, test_data, test_label = load_all_data_fashion_mnist(load_path, flatten=FLATTEN)
elif DATASET_MODE == 2:
    train_data, train_label, test_data, test_label = load_all_data_cifar10(load_path, flatten=FLATTEN)
elif DATASET_MODE == 3:
    train_data, train_label, test_data, test_label = load_all_data_apply_vgg_cifar10(load_path, flatten=FLATTEN)
else:
    train_data, train_label, test_data, test_label = None, None, None, None
train_data, train_label, test_data, test_label = create_batch_data(train_data, train_label, test_data, test_label,
                                                                   BATCH_SIZE)


# In[9]:


model_arr = [
    layers.RealPiNetSecondOrderConvLayer(3, 6, (9, 9)),
    modules.Flatten(),
    layers.RealPiNetSecondOrderLinearLayer(3456, 128),
    layers.RealLinearLayer(128, 10)
]

model = modules.Network(model_arr)
criterion = RealMSELoss()


# In[ ]:


for epoch in range(EPOCH):
    tot_loss = 0
    for train_idx, (train_data_batch, train_label_batch) in enumerate(zip(train_data, train_label)):
        # train
        preds = model.forward(train_data_batch)

        tot_loss += criterion.forward(preds, train_label_batch)
        propagated_error = criterion.error_derivative()

        model.backprop(propagated_error)
        model.optimize(LR)

        if train_idx == 0 or (train_idx + 1) % PRINT == 0:
            tot_acc = 0
            tot_sample = 0
            for train_acc_idx, (test_data_batch, test_label_batch) in enumerate(zip(test_data, test_label)):
                # train accuracy
                preds = model.forward(test_data_batch)
                pred_args = np.argmax(preds, axis=1)

                tot_acc += np.count_nonzero(pred_args == test_label_batch)
                tot_sample += test_data_batch.shape[0]
            accuracy = tot_acc / tot_sample
            if train_idx != 0:
                tot_loss = tot_loss / PRINT
            print('epoch: {}, idx: {}, accuracy: {}, loss: {}'.format(epoch + 1, train_idx + 1, accuracy, tot_loss))
            tot_loss = 0
