import argparse
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt

from simple_network_numpy import ScaledVectorizedIntegerNetNumpy, ScaledVectorizedFiniteFieldNetNumpy


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