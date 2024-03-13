import numpy as np
import galois
import sys
import logging
from datetime import datetime

from utils import create_batch_data

import nets.modules as modules

from nets.galois_net.datasets import load_all_data_cifar10_vgg
from nets.galois_net.utils import to_real_domain
import nets.galois_net.layers as layers
from nets.galois_net.activations import GaloisQuadraticActivation
from nets.galois_net.criterions import GaloisFieldMSELoss

now = datetime.now().strftime('%m%d%Y%H%M')
log_file_name = now + '_' + sys.argv[0].split('.')[0] + '.log'
logging.basicConfig(filename=log_file_name, filemode='w', format='[%(asctime)s] %(levelname)s - %(message)s', level=logging.DEBUG)


BATCH_SIZE = 256
EPOCH = 1
DATASET_USED = 'CIFAR-10-VGG'

QUANTIZATION_INPUT = 8
QUANTIZATION_WEIGHT = 32
QUANTIZATION_BATCH_SIZE = 8
LR = 7
PRIME = 136759815150493740654140208079

logging.info('###### EXPERIMENT DETAILS ######' +
             '\n\tBATCH SIZE: {}'.format(BATCH_SIZE) +
             '\n\tMAX NUMBER OF EPOCHS: {}'.format(EPOCH) +
             '\n\tINITIAL LR: {}'.format(LR) +
             '\n\tQUANTIZATION_INPUT: {}'.format(QUANTIZATION_INPUT) +
             '\n\tQUANTIZATION_WEIGHT: {}'.format(QUANTIZATION_WEIGHT) +
             '\n\tQUANTIZATION_BATCH_SIZE: {}'.format(QUANTIZATION_BATCH_SIZE) +
             '\n\tPRIME: {}'.format(PRIME) +
             '\n###### EXPERIMENT DETAILS ######\n')

field = galois.GF(PRIME)

# data fetching
load_path = './data'
train_data, train_label, test_data, test_label = load_all_data_cifar10_vgg(load_path, QUANTIZATION_INPUT, QUANTIZATION_WEIGHT, PRIME, field)
train_data, train_label, test_data, test_label = create_batch_data(train_data, train_label, test_data, test_label, BATCH_SIZE)

logging.info('NUMBER OF ITERATIONS IN ONE TRAINING EPOCH: {}\n'.format(len(train_data)))

model_arr = [
    layers.GaloisFieldLinearLayer(25088, 128, QUANTIZATION_WEIGHT, PRIME, field, first_layer=True, quantization_bit_input=QUANTIZATION_INPUT),
    GaloisQuadraticActivation(QUANTIZATION_WEIGHT, PRIME, field),
    layers.GaloisFieldLinearLayer(128, 10, QUANTIZATION_WEIGHT, PRIME, field)
]

model = modules.Network(model_arr)
criterion = GaloisFieldMSELoss(PRIME, QUANTIZATION_WEIGHT, QUANTIZATION_BATCH_SIZE, field)

for epoch in range(EPOCH):
    for train_idx, (train_data_batch, train_label_batch) in enumerate(zip(train_data, train_label)):
        # train
        preds = model.forward(train_data_batch)
        curr_training_loss = criterion.forward(preds, train_label_batch)
        propagated_error = criterion.error_derivative()

        model.backprop(propagated_error)
        model.optimize(LR)

        print('epoch: {}/{}, iteration: {}/{}, loss: {}'.format(epoch + 1, EPOCH, train_idx + 1, len(train_data), curr_training_loss))
        logging.info('epoch: {}/{}, iteration: {}/{}, loss: {}'.format(epoch + 1, EPOCH, train_idx + 1, len(train_data), curr_training_loss))

        if train_idx == 0 or (train_idx + 1) % 10 == 0 or train_idx == (len(train_data) - 1):
            tot_acc = 0
            tot_sample = 0
            for test_idx, (test_data_batch, test_label_batch) in enumerate(zip(test_data, test_label)):
                # train accuracy
                preds = model.forward(test_data_batch)
                preds = to_real_domain(preds, QUANTIZATION_WEIGHT, PRIME)
                pred_args = np.argmax(preds, axis=1)

                tot_acc += np.count_nonzero(pred_args == test_label_batch)
                tot_sample += test_data_batch.shape[0]
            
            accuracy = tot_acc / tot_sample
            print('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))
            logging.info('epoch: {}/{}, accuracy: {}'.format(epoch + 1, EPOCH, accuracy))
            