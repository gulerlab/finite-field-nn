import argparse
import logging
from datetime import datetime
import os
import numpy as np

from ff_clover import FiniteFieldClover


def save_params(mode, experiment_datetime, model):
    save_path = os.path.join('./params', '{}{}{}-{}{}{}-{}-ff-clover'.format(experiment_datetime.year,
                                                                             experiment_datetime.month,
                                                                             experiment_datetime.day,
                                                                             experiment_datetime.hour,
                                                                             experiment_datetime.minute,
                                                                             experiment_datetime.second, mode))
    os.makedirs(save_path)

    with open(os.path.join(save_path, 'running_loss.npy'), 'wb') as fp:
        # noinspection PyTypeChecker
        np.save(fp, model.running_loss)

    with open(os.path.join(save_path, 'running_acc.npy'), 'wb') as fp:
        # noinspection PyTypeChecker
        np.save(fp, model.running_acc)

    with open(os.path.join(save_path, 'running_curr_loss.npy'), 'wb') as fp:
        # noinspection PyTypeChecker
        np.save(fp, model.running_curr_loss)

    with open(os.path.join(save_path, 'weight_1.npy'), 'wb') as fp:
        # noinspection PyTypeChecker
        np.save(fp, model.weight_1)

    with open(os.path.join(save_path, 'weight_2.npy'), 'wb') as fp:
        # noinspection PyTypeChecker
        np.save(fp, model.weight_2)

    with open(os.path.join(save_path, 'elapsed_time.txt'), 'w') as fp:
        fp.write('{} seconds'.format(model.elapsed_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='experiment type')
    parser.add_argument('-p', '--prime', dest='prime', type=int, default=((2**26) - 5))
    parser.add_argument('-g', '--gamma', dest='gamma', type=int, default=8)
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=int, default=10)
    parser.add_argument('-sg', '--second-gamma', dest='second_gamma', type=int)
    args = parser.parse_args()

    experiment_now = datetime.now()
    year, month, day, hour, minute, second = experiment_now.year, experiment_now.month, experiment_now.day, \
        experiment_now.hour, experiment_now.minute, experiment_now.second

    log_file_name = '{}{}{}-{}{}{}-{}-ff-clover.log'.format(experiment_now.year, experiment_now.month,
                                                            experiment_now.day, experiment_now.hour,
                                                            experiment_now.minute, experiment_now.second, args.mode)
    log_file_path = os.path.join('./logs', log_file_name)
    logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG)
    logging.info('PRIME: {}'.format(args.prime))
    logging.info('GAMMA: {}'.format(args.gamma))
    if args.second_gamma:
        logging.info('WEIGHT GAMMA: {}'.format(args.second_gamma))
        second_gamma = args.secong_gamma
    else:
        second_gamma = args.gamma
    logging.info('LEARNING RATE: {}'.format(args.learning_rate))
    net = None
    if args.mode == 'fashion-mnist':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 784, 128, 10
        num_epoch, batch_size = 1, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train(num_epoch, batch_size)
    elif args.mode == 'mnist':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 784, 128, 10
        num_epoch, batch_size = 1, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train_mnist(num_epoch, batch_size)
    elif args.mode == 'mnist-v2':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 784, 128, 10
        num_epoch, batch_size = 300, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train_mnist_v2(num_epoch, batch_size)
    elif args.mode == 'cifar10':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 3072, 128, 10
        num_epoch, batch_size = 1, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train_cifar10(num_epoch, batch_size)
    elif args.mode == 'cifar10-vgg':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 25088, 128, 10
        num_epoch, batch_size = 1, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train_vgg_cifar10(num_epoch, batch_size)
    elif args.mode == 'cifar10-vgg-v2':
        scaled_input, scale_weight, learning_rate, prime = args.gamma, second_gamma, args.learning_rate, args.prime
        feature_size, hidden_layer_size, num_class = 25088, 128, args.learning_rate
        num_epoch, batch_size = 300, 256
        net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                                feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
        net.train_vgg_cifar10_v2(num_epoch, batch_size)
    if net is not None:
        save_params(args.mode, experiment_now, net)
    else:
        print('network can not be defined')
