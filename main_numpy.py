import argparse
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

from simple_network_numpy import ScaledVectorizedIntegerNetNumpy, ScaledVectorizedFiniteFieldNetNumpy,\
    ScaledIntegerNetNumpy, ScaledFiniteFieldNetNumpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='mode for training using numpy')
    args = parser.parse_args()

    experiment_now = datetime.now()
    year, month, day, hour, minute, second = experiment_now.year, experiment_now.month, experiment_now.day,\
        experiment_now.hour, experiment_now.minute, experiment_now.second

    log_file_name = '{}{}{}-{}{}{}-{}.log'.format(experiment_now.year, experiment_now.month, experiment_now.day,
                                                  experiment_now.hour, experiment_now.minute, experiment_now.second,
                                                  args.mode)
    log_file_path = os.path.join('./logs', log_file_name)
    logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG)
    if args.mode == 'scaled-vectorized-int-numpy':
        model = ScaledVectorizedIntegerNetNumpy(8, 8, 10)
        model.train('./data', 1, 0.001)
        running_acc = model.running_acc
        running_loss = model.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - integer net - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_int_net_np.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - integer net - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_int_net_np.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-vectorized-ff-numpy':
        model = ScaledVectorizedFiniteFieldNetNumpy(8, 8, 10, 2**26 - 5)
        model.train('./data', 1, 0.001)
    elif args.mode == 'scaled-int-numpy':
        model = ScaledIntegerNetNumpy(8, 8, 7)
        model.train('./data', 1, 0.01, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - integer net (minibatch) - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_int_net_minibatch_np.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - integer net (minibatch) - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_int_net_minibatch_np.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-ff-numpy':
        model = ScaledFiniteFieldNetNumpy(8, 8, 7, 2**26 - 5)
        model.train('./data', 1, 0.01, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - finite field net (minibatch) - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_ff_net_minibatch_np.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - finite field net (minibatch) - NumPy')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_ff_net_minibatch_np.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-int-numpy-cifar10':
        model = ScaledIntegerNetNumpy(8, 8, 10, input_vector_size=3072, hidden_layer_size=256)
        model.train_cifar10('./data', 1, 0.001, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - integer net (minibatch) - NumPy - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_int_net_minibatch_np_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - integer net (minibatch) - NumPy - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_int_net_minibatch_np_cifar10.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-ff-numpy-cifar10':
        model = ScaledFiniteFieldNetNumpy(8, 8, 10, 2**26 - 5, input_vector_size=3072, hidden_layer_size=256)
        model.train_cifar10(1, 0.001, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - finite field net (minibatch) - NumPy - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_ff_net_minibatch_np_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - finite field net (minibatch) - NumPy - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_ff_net_minibatch_np_cifar10.jpeg', dpi=300)
        plt.show()
    elif args.mode == 'scaled-int-numpy-vgg-cifar10':
        model = ScaledIntegerNetNumpy(8, 8, 10, input_vector_size=25088, hidden_layer_size=128)
        model.train_vgg_cifar10('./data', 1, 0.001, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        running_curr_loss = model.running_curr_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - integer net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_int_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_curr_loss)), running_curr_loss)
        plt.title('all loss vs. iteration - integer net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('all loss')
        plt.savefig('all_loss_int_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - integer net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_int_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()

        # noinspection DuplicatedCode
        save_path = os.path.join('./params', '{}{}{}-{}{}{}-{}'.format(experiment_now.year, experiment_now.month,
                                                                       experiment_now.day, experiment_now.hour,
                                                                       experiment_now.minute,
                                                                       experiment_now.second, args.mode))
        os.makedirs(save_path)

        with open(os.path.join(save_path, 'running_loss.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_loss)

        with open(os.path.join(save_path, 'running_acc.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_acc)

        with open(os.path.join(save_path, 'running_curr_loss.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_curr_loss)

        with open(os.path.join(save_path, 'weight_1.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, model.weight_1)

        with open(os.path.join(save_path, 'weight_2.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, model.weight_2)
    elif args.mode == 'scaled-ff-numpy-vgg-cifar10':
        model = ScaledFiniteFieldNetNumpy(8, 8, 10, 2**26 - 5, input_vector_size=25088, hidden_layer_size=128)
        model.train_vgg_cifar10(1, 0.001, 128)
        running_acc = model.running_acc
        running_loss = model.running_loss
        running_curr_loss = model.running_curr_loss
        plt.figure()
        plt.plot(range(len(running_loss)), running_loss)
        plt.title('loss vs. iteration - finite field net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig('loss_ff_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_curr_loss)), running_curr_loss)
        plt.title('all loss vs. iteration - finite field net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('all loss')
        plt.savefig('all_loss_ff_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()
        plt.plot(range(len(running_acc)), running_acc)
        plt.title('acc vs. iteration - finite field net (minibatch) - NumPy - VGG - CIFAR10')
        plt.xlabel('iteration')
        plt.ylabel('acc')
        plt.savefig('acc_ff_net_minibatch_np_vgg_cifar10.jpeg', dpi=300)
        plt.show()
        plt.figure()

        # noinspection DuplicatedCode
        save_path = os.path.join('./params', '{}{}{}-{}{}{}-{}'.format(experiment_now.year, experiment_now.month,
                                                                       experiment_now.day, experiment_now.hour,
                                                                       experiment_now.minute,
                                                                       experiment_now.second, args.mode))
        os.makedirs(save_path)

        with open(os.path.join(save_path, 'running_loss.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_loss)

        with open(os.path.join(save_path, 'running_acc.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_acc)

        with open(os.path.join(save_path, 'running_curr_loss.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, running_curr_loss)

        with open(os.path.join(save_path, 'weight_1.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, model.weight_1)

        with open(os.path.join(save_path, 'weight_2.npy'), 'wb') as fp:
            # noinspection PyTypeChecker
            np.save(fp, model.weight_2)
