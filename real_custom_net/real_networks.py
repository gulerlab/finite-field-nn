from real_module import RealModule
from real_layers import RealLinearLayer
from real_activations import RealQuadraticAct


class RealNetworkLinearTest(RealModule):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealLinearLayer(784, 128),
            RealQuadraticAct(),
            RealLinearLayer(128, 10)
        ]

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self.__model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self.__model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self.__model:
            if isinstance(layer, RealModule):
                layer.optimize(learning_rate)

    def loss(self):
        pass


class RealNetworkLinearTest_v2(RealModule):
    def __init__(self):
        super().__init__()
        self.__model = [
            RealLinearLayer(784, 512),
            RealQuadraticAct(),
            RealLinearLayer(512, 256),
            RealQuadraticAct(),
            RealLinearLayer(256, 128),
            RealQuadraticAct(),
            RealLinearLayer(128, 64),
            RealQuadraticAct(),
            RealLinearLayer(64, 32),
            RealQuadraticAct(),
            RealLinearLayer(32, 10),
        ]

    def forward(self, input_data):
        self._input_data = input_data
        curr_data = self._input_data
        for layer in self.__model:
            curr_data = layer.forward(curr_data)
        return curr_data

    def backprop(self, propagated_error):
        self._propagated_error = propagated_error
        curr_error = self._propagated_error
        for layer in reversed(self.__model):
            layer.backprop(curr_error)
            curr_error = layer.loss()

    def optimize(self, learning_rate):
        for layer in self.__model:
            if isinstance(layer, RealModule):
                layer.optimize(learning_rate)

    def loss(self):
        pass
