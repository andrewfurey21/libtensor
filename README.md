# libtensor

<!--  TODO: Show image of generated graph + mnist example  -->

Zero-dependency tensor library with automatic differentiation for deep learning.

Does need GoogleTest for testing though.

## current features

- [x] tensor
- [x] backwards mode autodifferentiation
- [ ] ops
    - [x] add/sub, mul
    - [x] square, sqrt, log, exp
    - [x] reshape, flatten (just a reshape)
    - [x] sum/expand along axis
    - [x] relu
    - [ ] matmul
    - [x] max pool
    - [x] convolutions
    - [x] sparse categorical cross entropy
- [ ] sgd optimizer

## notes

Here's a mini example of working with tensors to compute gradients.

```c
ttuple* input_shape = ttuple_build(2, 4, 6); // shape (4, 6)
tt* a = tt_uniform(input_shape, -10, 10, true);
tt* b = tt_uniform(input_shape, -10, 10, true);
tt* a_b_mul = tt_mul(a, b);
tt* sum = tt_sum(a_b_mul, -1);
tgraph* comp_graph = tgraph_build(sum);
tgraph_zeroed(comp_graph);
tgraph_backprop(comp_graph);
```

## examples

- [ ] mnist cnn

## future ideas

- broadcasting, keepdim, proper views, strides, proper storage abstraction, contiguous
- better viz of graph
- could totally do a refactor, might be nice to have a context like ggml. make it so that memory doesnt get reallocated when running.
- coreops.c, ops.c, storage.c, graph.c, nn.c (neural net specific likes optimizers), extra (tuples etc)
- need graph.c?
- could do permute+pad op, then redo maxpools/convs.
- groups, stride, dilation, padding to maxpool + conv. Figure out how im2col works, and how tinygrad does it's convolutions
- more functions: batchnorm, attention,, etc.
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
