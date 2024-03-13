import numpy as np
import sys
import logging
from datetime import datetime

from utils import create_batch_data

import nets.modules as modules

from nets.int_net.datasets import load_all_data_mnist, load_all_data_cifar10, load_all_data_fashion_mnist
from nets.int_net.utils import to_real_domain
import nets.int_net.layers as layers
from nets.int_net.activations import GAPTruncation
from nets.int_net.criterions import IntMSELoss
import nets.int_net.minmax_cache as minmax_cache

now = datetime.now().strftime('%m%d%Y%H%M')
log_file_name = now + '_' + sys.argv[0].split('.')[0] + '_' + sys.argv[1] + '.log'
logging.basicConfig(filename=log_file_name, filemode='w', format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.INFO)

BATCH_SIZE = 16
EPOCH = 50
FLATTEN = False
# 0, MNIST; 1, FashionMNIST; 2, CIFAR10
DATASET_MODE = 2
DATASET_USED = ['MNIST', 'FashionMNIST', 'CIFAR-10']

QUANTIZATION_INPUT = 8
QUANTIZATION_WEIGHT = 24
QUANTIZATION_BATCH_SIZE = 4
LR = 7
PRIME = int(sys.argv[1])

minmax_cache.init_globals(EPOCH)
minmax_cache.prime = PRIME

logging.info('###### EXPERIMENT DETAILS ######' +
             '\n\tBATCH SIZE: {}'.format(BATCH_SIZE) +
             '\n\tMAX NUMBER OF EPOCHS: {}'.format(EPOCH) +
             '\n\tINITIAL LR: {}'.format(LR) +
             '\n\tFLATTEN: {}'.format(FLATTEN) +
             '\n\tDATASET USED: {}'.format(DATASET_USED[DATASET_MODE]) +
             '\n\tQUANTIZATION_INPUT: {}'.format(QUANTIZATION_INPUT) +
             '\n\tQUANTIZATION_WEIGHT: {}'.format(QUANTIZATION_WEIGHT) +
             '\n\tQUANTIZATION_BATCH_SIZE: {}'.format(QUANTIZATION_BATCH_SIZE) +
             '\n\tPRIME: {}'.format(PRIME) +
             '\n###### EXPERIMENT DETAILS ######\n')

# data fetching
load_path = './data'
if DATASET_MODE == 0:
    train_data, train_label, test_data, test_label = load_all_data_mnist(load_path, QUANTIZATION_INPUT, QUANTIZATION_WEIGHT, flatten=FLATTEN)
elif DATASET_MODE == 1:
    train_data, train_label, test_data, test_label = load_all_data_fashion_mnist(load_path, QUANTIZATION_INPUT, QUANTIZATION_WEIGHT, flatten=FLATTEN)
elif DATASET_MODE == 2:
    train_data, train_label, test_data, test_label = load_all_data_cifar10(load_path, QUANTIZATION_INPUT, QUANTIZATION_WEIGHT, flatten=FLATTEN)
else:
    train_data, train_label, test_data, test_label = None, None, None, None
train_data, train_label, test_data, test_label = create_batch_data(train_data, train_label, test_data, test_label, BATCH_SIZE)

logging.info('NUMBER OF ITERATIONS IN ONE TRAINING EPOCH: {}\n'.format(len(train_data)))

model_arr = [
    layers.IntPiNetSecondOrderConvLayer(3, 64, (4, 4), QUANTIZATION_WEIGHT, first_layer=True,
                                                quantization_bit_input=QUANTIZATION_INPUT, stride=(2, 2), padding=(1, 1, 1, 1)),
    layers.IntPiNetSecondOrderConvLayer(64, 256, (4, 4), QUANTIZATION_WEIGHT, stride=(2, 2), padding=(1, 1, 1, 1)),
    GAPTruncation(),
    layers.IntLinearLayer(256, 10, QUANTIZATION_WEIGHT)
]

model = modules.Network(model_arr)
criterion = IntMSELoss(QUANTIZATION_WEIGHT, QUANTIZATION_BATCH_SIZE)

for epoch in range(EPOCH):
    minmax_cache.epoch = epoch
    for train_idx, (train_data_batch, train_label_batch) in enumerate(zip(train_data, train_label)):
        minmax_cache.overflow_monitor = 0
        # train
        preds = model.forward(train_data_batch)
        curr_training_loss = criterion.forward(preds, train_label_batch)
        propagated_error = criterion.error_derivative()

        model.backprop(propagated_error)
        model.optimize(LR)

        print('epoch: {}/{}, iteration: {}/{}, loss: {}'.format(epoch + 1, EPOCH, train_idx + 1, len(train_data), curr_training_loss))
        logging.info('epoch: {}/{}, iteration: {}/{}, loss: {}, for {} number of parameters overflow occured'.format(epoch + 1, EPOCH, train_idx + 1,
                                                                                                                     len(train_data),
                                                                                                                     curr_training_loss,
                                                                                                                     minmax_cache.overflow_monitor))

    tot_acc = 0
    tot_sample = 0
    for test_idx, (test_data_batch, test_label_batch) in enumerate(zip(test_data, test_label)):
        # train accuracy
        preds = model.forward(test_data_batch)
        preds = to_real_domain(preds, QUANTIZATION_WEIGHT)
        pred_args = np.argmax(preds, axis=1)

        tot_acc += np.count_nonzero(pred_args == test_label_batch)
        tot_sample += test_data_batch.shape[0]
    
    accuracy = tot_acc / tot_sample
    print('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))
    logging.info('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))
    logging.info('epoch: {}/{}, min: {}, max: {}'.format(epoch + 1, EPOCH, minmax_cache.min_value[epoch], minmax_cache.max_value[epoch]))
            