import torch
from torch.nn import functional as F
if __name__ == "__main__":
    input = torch.linspace(0, 5*5-1, 5*5).reshape(1, 1, 5, 5).requires_grad_(True) 
    print("Input requires grad: ", input.requires_grad)

    conv_weights = torch.linspace(0, 3*3-1, 3*3).reshape(1, 1, 3, 3).requires_grad_(True) 
    print("Conv weights grad: ", conv_weights.requires_grad)

    conv = F.conv2d(input, conv_weights)
    conv.retain_grad()

    input_sum = input.sum()
    input_sum.retain_grad()
    conv_sum = conv.sum()
    conv_sum.retain_grad()

    output = conv_sum + input_sum
    output.retain_grad()
    print("Output: \n", output)

    output.backward()

    print(conv.grad)
    print(input_sum.grad)

    print("Input grads: \n", input.grad)
