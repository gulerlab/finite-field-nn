import torch
import torch.nn as nn

class CustomNet():
    def __init__(self, input_vector, hidden_layer_size=64, num_classes=10, device=None) -> None:
        super.__init__()
        self.input_vector = input_vector
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes
        self.__init_weight()

    def __init_weight(self):
        self.weight_1 = torch.empty(self.input_vector.size(0), self.hidden_layer_size)
        self.weight_2 = torch.empty(self.hidden_layer_size, self.num_classes)
        nn.init.kaiming_normal_(self.weight_1)
        nn.init.kaiming_normal_(self.weight_2)
    
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