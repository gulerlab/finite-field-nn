import torch
import torch.nn as nn
from utils import to_real_domain, to_finite_field_domain, finite_field_truncation


class VectorizedScaledNet:
    def __init__(self, scale_x, scale_w, input_vector_size=784, hidden_layer_size=64, num_classes=10, device=None,
                 verbose=True) -> None:
        self.input_vector_size = input_vector_size
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes
        self.device = device
        self.verbose = verbose

        self.__to_device()
        self.__init_weight()

    def __init_weight(self):
        self.weight_1 = torch.empty(self.input_vector_size, self.hidden_layer_size)
        self.weight_2 = torch.empty(self.hidden_layer_size, self.num_classes)
        nn.init.kaiming_normal_(self.weight_1)
        nn.init.kaiming_normal_(self.weight_2)

    def __to_device(self):
        self.weight_1 = self.weight_1.to(self.device)
        self.weight_2 = self.weight_2.to(self.device)

        # if self.verbose:

    def __init_scale(self):
        pass

    def __criterion(self):
        pass

    def __optimizer(self):
        pass

    def __scaled_forward(self):
        pass

    def __scaled_backward(self):
        pass

    def train(self):
        pass
