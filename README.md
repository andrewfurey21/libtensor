# libtensor

<!--  TODO: Show image of generated graph + mnist example  -->

Zero-dependency tensor library with automatic differentiation for deep learning.

My approach was building a small set of base operations that can be used to build more complicated operations like matmul, convs etc, sort of like tinygrad.

## current features

- [x] tensor
- [x] backwards mode autodifferentiation
- [ ] ops
    - [x] add/sub, mul
    - [x] square, sqrt, log, exp
    - [x] reshape, flatten (just a reshape)
    - [x] sum/expand along axis
    - [x] relu
    - [x] matmul/dot (indirectly, using reshape, expand, mul and sum)
    - [x] max pool
    - [x] convolutions
    - [x] sparse categorical cross entropy
- [ ] sgd optimizer
- [ ] import/export weights

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

### how are matmul, batch norms and convolutions done?

There is no direct matmul op. You have to do it manually with a reshape, expand, mul and sum.

<!-- TODO: show image from excalidraw doing matmul -->

Once I implement permute, it should just be transpose -> mul -> sum.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of matmul -->

There is also no direct batch norm op. You have to do it manually.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of batch norm -->

Convolutions are implemented as their on kernel for the moment.

## examples

- [ ] mnist handwritten digit recognition

## future ideas

- broadcasting, keepdim, proper views, strides, proper storage abstraction (like numpy)
- better viz
- could totally do a refactor, might be nice to have a context like ggml. make it so that memory doesnt get allocated when running.
- coreops.c, ops.c, storage.c, graph.c, nn.c (neural net specific likes optimizers), extra (tuples etc)
- need graph.c?
- could do permute+pad op, then redo maxpools/convs.
- more ops: batchnorm, attention, etc.
- different backends (opencl/vulkan, cuda, metal, avx/sse, triton, rocm, tenstorrent)
- more example models (yolo, gpt, sam etc)
- choose different types (double, f16, bfloat, mx-compliant)
- other optimizer implementations (adagrad, rmsprop, adam, demo, etc)
- other convolution implementations (singular value decomposition, FFT, winograd)
- python bindings
- multigpu
- onnx support
