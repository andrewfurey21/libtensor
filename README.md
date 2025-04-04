# newmind

<!--  TODO: Show gif of it solving mnist and generated graph -->

Zero-dependency tensor library with automatic differentiation for deep learning.

Does need GoogleTest for to run tests though.

## features

- [x] tensor
- [x] backwards mode autodiff
- [ ] sgd optimizer
- [ ] avx/openmp backend
- [ ] cuda backend

## examples

- [ ] mnist cnn

## notes

Here's a mini example of working with tensors to compute gradients.

```c

#include "../include/tensor.h"

int main(void) {
  intarray *ashape = intarray_build(2, 4, 6); // shape (4, 6)
  tensor *a = tensor_linspace(ashape, -10, 10, true);

  intarray *bshape = intarray_build(2, 6, 5); // shape (6, 5)
  tensor *b = tensor_linspace(bshape, -10, 10, true);

  tensor *matmul = tensor_matmul(a, b, true);
  tensor *sum = tensor_sum(matmul, -1, true);

  graph *network = graph_build(sum);
  graph_zeroed(network);
  graph_backprop(network);

  graph_print(network, true, true);

  intarray_free(ashape);
  intarray_free(bshape);
  graph_free(network);
}

```

## future ideas

- broadcasting, in place ops, reduce mallocs by reusing allocs. vectorize/gpu. minimize/optimize cache usage. allocate the graph once, do stuff lazy. have forward function in the tensor or something similar to backward pass. custom allocator for this. that and kernels will be part of backend maybe. read about tcmalloc/jemalloc/malloc/pytorch cuda alloc, maybe a more generic version of this?
- ggml kind of context? lazy setup tensors. no mallocs during inference/training
- groups, stride, dilation, padding to maxpool + conv. Figure out how im2col works, and how tinygrad does it's convolutions
- more functions: permute, pad, batchnorm, attention, etc.
- import/export weights in different formats
- proper backend system (opencl/vulkan, cuda, metal, avx/sse, triton, rocm, tenstorrent)
- more example models (yolo, gpt, sam etc)
- choose different types (double, f16, bfloat, mx-compliant)
- other optimizer implementations (adagrad, rmsprop, adam, demo, etc)
- other convolution implementations (singular value decomposition, FFT, winograd)
- python bindings
- multigpu
- onnx support
- homomorphic encryption
- quantization
