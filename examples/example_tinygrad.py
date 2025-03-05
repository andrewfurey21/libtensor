from tinygrad import Tensor, nn
from functools import reduce
import csv

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def mult(i): return reduce(lambda x, y: x * y, i)

class Model:
    def __init__(self):
        self.layers = [
            nn.Conv2d(1, 32, 3), Tensor.relu,
            nn.Conv2d(32, 64, 3), Tensor.relu,
            Tensor.max_pool2d, lambda x: x.flatten(1),
            nn.Linear(9216, 128), Tensor.relu,
            nn.Linear(128, 10), Tensor.log_softmax
        ]

    def __call__(self, x:Tensor) -> Tensor:
        return x.sequential(self.layers)

def load_mnist_batch(file_name:str, line:int):
    with open(file_name, "r") as f:
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            if line == i:
                return row

def draw_mnist_image(data, size=(28, 28)):
    print(f"Batch #{1}:")
    for i in range(size[0]):
        for j in range(size[1]):
            value = float(data[j + i * size[0]])
            if value > 150:
                print("MM", end="")
            else:
                print("..", end="")
        print("")


if __name__ == "__main__":
    # ???
    d = Tensor.linspace(0, 5*5-1, 5*5).reshape(1, 1, 5, 5).requires_grad_(True)
    print("Input weights grad: ", d.requires_grad)
    conv_weights = Tensor.linspace(0, 3*3-1, 3*3).reshape(1, 1, 3, 3).requires_grad_(True)
    print("Conv weights grad: ", conv_weights.requires_grad)

    conv = d.conv2d(conv_weights)

    input_sum = d.sum()
    conv_sum = conv.sum()

    output = conv_sum #+ input_sum

    print(output.numpy())
    output.backward()


    print(d.grad.numpy())
    print(conv.grad)

    a = Tensor.linspace(0, 5, 6).reshape(2, 3).requires_grad_(True)
    b = Tensor.linspace(0, 11, 12).reshape(2, 3, 2).requires_grad_(True)
    c = a.matmul(b)
    s = c.sum()
    s.backward()
    #
    # print("input: \n", b.numpy())
    print("input.grad: \n", b.grad.numpy())
    #
    # print("weights: \n", a.numpy())
    # print("weights.grad: \n", a.grad.numpy())
    #
    # print("matmul: \n", c.numpy())
    # print("matmul.grad: \n", c.grad.numpy())
    # model = Model()

    # example = 123
    # data = load_mnist_batch("./data/mnist_test.csv", example)[1:]
    # draw_mnist_image(data);
    # float_data = [float(d) for d in data]
    # input_batch = Tensor(float_data).reshape((1, 1, 28, 28))
    # output = model(input_batch)
    # print(output.numpy())
    #
    # # conv_1_shape   = (32, 1, 3, 3)
    # # conv_1_weights = Tensor.linspace(0, 10, mult(conv_1_shape)).reshape(conv_1_shape)
    # #
    # # layer_1 = input_batch.conv2d(conv_1_weights)
    #
    # ex = Tensor.linspace(0, 10, 12, requires_grad=True).reshape(12)
    # lsm = ex.log_softmax()
    # print(lsm.numpy())
    # s = lsm.sparse_categorical_crossentropy(Tensor([1]))
    # s.backward()
    # print(ex.grad.numpy())







    


    
