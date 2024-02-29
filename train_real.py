import numpy as np
from nets.real_net.datasets import load_all_data_mnist, load_all_data_cifar10, load_all_data_fashion_mnist, load_all_data_apply_vgg_cifar10
from utils import create_batch_data
import modules
import nets.real_net.layers as layers
from nets.real_net.activations import GAPTruncation
from nets.real_net.criterions import RealMSELoss

import sys
import logging
from datetime import datetime
import os
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--cp', '-checkpoint-path', dest='checkpoint_path', type=str)
args = parser.parse_args()

now = datetime.now().strftime('%m%d%Y%H%M')
log_file_name = now + '_' + sys.argv[0].split('.')[0] + '.log'
logging.basicConfig(filename=log_file_name, filemode='w', format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)

BATCH_SIZE = 16
EPOCH = 500
LR = 0.01
PRINT = 10
FLATTEN = False
# 0, MNIST; 1, FashionMNIST; 2, CIFAR10; 3, VGG-CIFAR10
DATASET_MODE = 2
DATASET_USED = ['MNIST', 'FashionMNIST', 'CIFAR-10', 'VGG-CIFAR-10']
SCHEDULE_LR = [200, 400]
STARTING_EPOCH = 0
SCALE_LOADED_LR = True

# TODO: for now save in this way, automize this line as well
if args.checkpoint_path is None:
    save_name = now + '_' + sys.argv[0].split('.')[0] + '.pkl'
    SAVE_PATH = './checkpoints/{}'.format(save_name)
else:
    SAVE_PATH = args.checkpoint_path

checkpoint = None
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, 'rb') as fp:
        checkpoint = pickle.load(fp)
    STARTING_EPOCH = checkpoint['epoch']
    LR = checkpoint['lr']
    if SCALE_LOADED_LR:
        LR = LR * 0.1

logging.info('###### EXPERIMENT DETAILS ######' +
             '\n\tBATCH SIZE: {}'.format(BATCH_SIZE) +
             '\n\tMAX NUMBER OF EPOCHS: {}'.format(EPOCH) +
             '\n\tINITIAL LR: {}'.format(LR) +
             '\n\tFLATTEN: {}'.format(FLATTEN) +
             '\n\tDATASET USED: {}'.format(DATASET_USED[DATASET_MODE]) +
             '\n\tSCHEDULE_LR: {}'.format(SCHEDULE_LR) +
             '###### EXPERIMENT DETAILS ######\n')

# data fetching
load_path = './data'
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
train_data, train_label, test_data, test_label = create_batch_data(train_data, train_label, test_data, test_label, BATCH_SIZE)

logging.info('NUMBER OF ITERATIONS IN ONE TRAINING EPOCH: {}\n'.format(len(train_data)))

model_arr = [
    layers.RealPiNetSecondOrderConvLayer(3, 64, (4, 4), stride=(2, 2), padding=(1, 1, 1, 1)),
    layers.RealPiNetSecondOrderConvLayer(64, 256, (4, 4), stride=(2, 2), padding=(1, 1, 1, 1)),
    GAPTruncation(),
    layers.RealLinearLayer(256, 10)
]

model = modules.Network(model_arr)
if checkpoint is not None:
    model.load_all_weights(checkpoint['state_dict'])

criterion = RealMSELoss()

for epoch in range(EPOCH):
    if (epoch + 1) in SCHEDULE_LR:
        LR = LR * 0.1
        print('LR SET TO {}'.format(LR))
        logging.info('LR SET TO {}'.format(LR))

    for train_idx, (train_data_batch, train_label_batch) in enumerate(zip(train_data, train_label)):
        # train
        preds = model.forward(train_data_batch)
        curr_training_loss = criterion.forward(preds, train_label_batch)
        propagated_error = criterion.error_derivative()

        model.backprop(propagated_error)
        model.optimize(LR)

        print('epoch: {}/{}, iteration: {}/{}, loss: {}'.format(epoch + 1, EPOCH, train_idx + 1, len(train_data), curr_training_loss))
        logging.info('epoch: {}/{}, iteration: {}/{}, loss: {}'.format(epoch + 1, EPOCH, train_idx + 1, len(train_data), curr_training_loss))

        
    tot_acc = 0
    tot_sample = 0
    for test_idx, (test_data_batch, test_label_batch) in enumerate(zip(test_data, test_label)):
        # train accuracy
        preds = model.forward(test_data_batch)
        pred_args = np.argmax(preds, axis=1)

        tot_acc += np.count_nonzero(pred_args == test_label_batch)
        tot_sample += test_data_batch.shape[0]
    
    accuracy = tot_acc / tot_sample
    print('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))
    logging.info('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))

    state_dict = model.return_all_weights()
    
    if not os.path.exists(os.path.split(SAVE_PATH)[0]):
        os.makedirs(os.path.split(SAVE_PATH)[0])

    checkpoint_dict = {
        'state_dict': state_dict,
        'epoch': epoch + 1,
        'lr': LR
    }
    with open(SAVE_PATH, 'wb') as fp:
        pickle.dump(checkpoint_dict, fp, pickle.HIGHEST_PROTOCOL)

    logging.info('model is saved')