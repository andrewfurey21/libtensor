from functools import reduce
import torch
from operator import mul

def prod(shape): return reduce(mul, shape)

input_shape = (1, 1, 28, 28)
input = torch.linspace(-10, 10, steps=prod(input_shape), dtype=torch.float32).reshape(input_shape).requires_grad_()

#################################

conv_layer_1_shape = (32, 1, 3, 3)
conv_layer_1 = torch.linspace(-10, 10, steps=prod(conv_layer_1_shape)).reshape(conv_layer_1_shape).requires_grad_()

l1 = torch.nn.functional.conv2d(input, conv_layer_1)
l1_activations = l1.relu()

#################################

conv_layer_2_shape = (64, 32, 3, 3)
conv_layer_2 = torch.linspace(-10, 10, steps=prod(conv_layer_2_shape)).reshape(conv_layer_2_shape).requires_grad_()

l2 = torch.nn.functional.conv2d(l1_activations, conv_layer_2)
l2_activations = l2.relu()

#################################

maxpool = torch.nn.functional.max_pool2d(l2_activations, kernel_size=(2, 2))
flattened_maxpool = maxpool.flatten()

#################################

linear_layer_1_shape = (128, 9216)
linear_layer_1 = torch.linspace(-10, 10, steps=prod(linear_layer_1_shape)).reshape(linear_layer_1_shape).requires_grad_()

l3 = linear_layer_1.matmul(flattened_maxpool)
l3_activations = l3.relu()

#################################

linear_layer_2_shape = (10, 128)
linear_layer_2 = torch.linspace(-10, 10, steps=prod(linear_layer_2_shape)).reshape(linear_layer_2_shape).requires_grad_()
logits = linear_layer_2.matmul(l3_activations)

#################################

log_probs = torch.nn.functional.log_softmax(logits, dim=0)
output = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
loss = torch.nn.functional.cross_entropy(log_probs, output)
print(loss.tolist())

loss.backward()
print(input.grad.tolist())



