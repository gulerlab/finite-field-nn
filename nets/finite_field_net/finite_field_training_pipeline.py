#!/usr/bin/env python
# coding: utf-8

# imports

import sys
sys.path.append('../../')

import numpy as np
from datasets import load_all_data_mnist, load_all_data_cifar10, load_all_data_fashion_mnist,\
    load_all_data_apply_vgg_cifar10
from utils import create_batch_data
import modules
import layers
from criterions import FiniteFieldMSELoss

import argparse
import yaml

# argument parser
parser = argparse.ArgumentParser()

# required
parser.add_argument('-bs', '--batch_size', type=int, required=True, help='batch size')
parser.add_argument('-e', '--epoch', type=int, required=True, help='number of epochs')
parser.add_argument('-dm', '--dataset_mode', type=int, required=True, choices=range(4), help='dataset mode\n\t0:'
                                                                                             ' MNIST\n\t1: FashionMNIST'
                                                                                             '\n\t2: CIFAR10\n\t'
                                                                                             '3:VGG-CIFAR10')
parser.add_argument('-f', '--flatten', required=True, action='store_true', help='is flatten true or not')
parser.add_argument('-qi', '--quantization_input', required=True, type=int, help='quantization parameter for input')
parser.add_argument('-qw', '--quantization_weight', required=True, type=int, help='quantization parameter for weights')
parser.add_argument('-qbs', '--quantization_batch_size', required=True, type=int, help='quantization parameter'
                                                                                       ' for batch size')
parser.add_argument('-lr', '--learning_rate', required=True, type=int, help='learning rate for finite field experiment')
parser.add_argument('-p', '--prime', required=True, type=int, help='prime of the finite field')
parser.add_argument('-mcp', '--model_configuration_path', required=True, type=str, help='the path to the model'
                                                                                        ' configuration')

# additional
parser.add_argument('-p', '--print', type=int, default=10, help='print after each given number of iterations')
parser.add_argument('-dlp', '--data_load_path', default='../../data', type=str, help='data load path')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
EPOCH = args.epoch
PRINT = args.print
FLATTEN = args.flatten
# 0, MNIST; 1, FashionMNIST; 2, CIFAR10; 3, VGG-CIFAR10
DATASET_MODE = args.dataset_mode

QUANTIZATION_INPUT = args.quantization_input
QUANTIZATION_WEIGHT = args.quantization_weight
QUANTIZATION_BATCH_SIZE = args.quantization_batch_size
LR = args.learning_rate
PRIME = args.prime

# data fetching
load_path = args.data_load_path
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

# load model

with open(args.model_configuration_path, 'r') as fp:
    model_conf = yaml.safe_load(fp)

model_layers_conf = model_conf['model']
model_name = model_conf['name']

model_arr = []
for idx, key in enumerate(model_layers_conf.keys()):
    layer_conf = model_layers_conf[key]
    layer_type = layer_conf['layer_type']
    if layer_type == 'pinet':
        pinet_instance = layer_conf['pinet_instance']
        if pinet_instance == 'conv':
            conv_in = layer_conf['conv_in']
            conv_out = layer_conf['conv_out']
            conv_kernel = layer_conf['conv_kernel']
            if idx == 0:
                model_arr.append(layers.FiniteFieldPiNetSecondOrderConvLayer(conv_in, conv_out, conv_kernel,
                                                                             QUANTIZATION_WEIGHT, PRIME,
                                                                             first_layer=True,
                                                                             quantization_bit_input=QUANTIZATION_INPUT))
            else:
                model_arr.append(layers.FiniteFieldPiNetSecondOrderConvLayer(conv_in, conv_out, conv_kernel,
                                                                             QUANTIZATION_WEIGHT, PRIME))
        elif pinet_instance == 'linear':
            linear_in = layer_conf['linear_in']
            linear_out = layer_conf['linear_out']
            if idx == 0:
                model_arr.append(
                    layers.FiniteFieldPiNetSecondOrderLinearLayer(linear_in, linear_out,
                                                                  QUANTIZATION_WEIGHT, PRIME,
                                                                  first_layer=True,
                                                                  quantization_bit_input=QUANTIZATION_INPUT)
                )
            else:
                model_arr.append(layers.FiniteFieldPiNetSecondOrderLinearLayer(linear_in, linear_out,
                                                                               QUANTIZATION_WEIGHT, PRIME))
    elif layer_type == 'general':
        general_instance = layer_conf['general_instance']
        if general_instance == 'conv':
            conv_in = layer_conf['conv_in']
            conv_out = layer_conf['conv_out']
            conv_kernel = layer_conf['conv_kernel']
            if idx == 0:
                model_arr.append(layers.FiniteFieldConvLayer(conv_in, conv_out, conv_kernel,
                                                             QUANTIZATION_WEIGHT, PRIME,
                                                             first_layer=True,
                                                             quantization_bit_input=QUANTIZATION_INPUT))
            else:
                model_arr.append(layers.FiniteFieldConvLayer(conv_in, conv_out, conv_kernel,
                                                             QUANTIZATION_WEIGHT, PRIME))
        elif general_instance == 'linear':
            linear_in = layer_conf['linear_in']
            linear_out = layer_conf['linear_out']
            if idx == 0:
                model_arr.append(
                    layers.FiniteFieldLinearLayer(linear_in, linear_out,
                                                  QUANTIZATION_WEIGHT, PRIME,
                                                  first_layer=True,
                                                  quantization_bit_input=QUANTIZATION_INPUT)
                )
            else:
                model_arr.append(layers.FiniteFieldLinearLayer(linear_in, linear_out,
                                                               QUANTIZATION_WEIGHT, PRIME))
    elif layer_type == 'flatten':
        model_arr.append(modules.Flatten())

model = modules.Network(model_arr)
criterion = FiniteFieldMSELoss(PRIME, QUANTIZATION_WEIGHT, QUANTIZATION_BATCH_SIZE)

# training and testing loop

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
