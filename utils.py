import numpy as np
import logging

#############
# common dataset operations
#############
def create_batch_data(train_data, train_label, test_data, test_label, batch_size):
    train_num_samples, test_num_samples = train_data.shape[0], test_data.shape[0]
    number_of_full_batch_train = int(train_num_samples / batch_size)
    last_batch_size_train = train_num_samples % batch_size

    number_of_full_batch_test = int(test_num_samples / batch_size)
    last_batch_size_test = test_num_samples % batch_size

    last_batch_train_data = None
    if last_batch_size_train != 0:
        last_batch_train_data = train_data[train_num_samples - last_batch_size_train:, :]

    if number_of_full_batch_train > 0:
        train_data = np.split(train_data[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    else:
        train_data = []
    if last_batch_train_data is not None:
        train_data.append(last_batch_train_data)

    last_batch_train_label = None
    if last_batch_size_train != 0:
        last_batch_train_label = train_label[train_num_samples - last_batch_size_train:, :]

    if number_of_full_batch_train > 0:
        train_label = np.split(train_label[:train_num_samples - last_batch_size_train, :], number_of_full_batch_train)
    else:
        train_label = []
    if last_batch_train_label is not None:
        train_label.append(last_batch_train_label)

    last_batch_test_data = None
    if last_batch_size_test != 0:
        last_batch_test_data = test_data[test_num_samples - last_batch_size_test:, :]

    if number_of_full_batch_test > 0:
        test_data = np.split(test_data[:test_num_samples - last_batch_size_test, :], number_of_full_batch_test)
    else:
        test_data = []
    if last_batch_test_data is not None:
        test_data.append(last_batch_test_data)

    last_batch_test_label = None
    if last_batch_size_test != 0:
        last_batch_test_label = test_label[test_num_samples - last_batch_size_test:]

    if number_of_full_batch_train > 0:
        test_label = np.split(test_label[:test_num_samples - last_batch_size_test], number_of_full_batch_test)
    else:
        test_label = []
    if last_batch_test_label is not None:
        test_label.append(last_batch_test_label)

    return train_data, train_label, test_data, test_label


#############
# utils for debug
#############

def info(msg, verbose=True):
    logging.info(msg)
    if verbose:
        print(msg)