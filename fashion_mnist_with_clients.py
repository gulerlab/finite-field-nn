import logging
from datetime import datetime
from ff_clover import FiniteFieldClover
import os

experiment_now = datetime.now()
year, month, day, hour, minute, second = experiment_now.year, experiment_now.month, experiment_now.day, \
    experiment_now.hour, experiment_now.minute, experiment_now.second

log_file_name = '{}{}{}-{}{}{}-{}-ff-clover.log'.format(experiment_now.year, experiment_now.month,
                                                        experiment_now.day, experiment_now.hour,
                                                        experiment_now.minute, experiment_now.second, 'fashion-mnist')
log_file_path = os.path.join('./logs', log_file_name)
logging.basicConfig(filename=log_file_path, encoding='utf-8', level=logging.DEBUG)

scaled_input, scale_weight, learning_rate, prime = 8, 8, 7, 2 ** 26 - 5
feature_size, hidden_layer_size, num_class = 784, 128, 10
num_epoch, batch_size = 1, 256
net = FiniteFieldClover(scaled_input, scale_weight, learning_rate, prime,
                        feature_size=feature_size, hidden_layer_size=hidden_layer_size, num_classes=num_class)
net.train(num_epoch, batch_size)