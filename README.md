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
    - [ ] matmul
    - [x] max pool
    - [x] convolutions
    - [x] sparse categorical cross entropy
- [ ] sgd optimizer

## todos

- tensor: fix flatten to work with matmul
- tensor: implement matmul backwards
- optimizers: reimplement sgd.
- get makefile to compile all programs into multiple binaries
- tensor: get rid of storage abstraction and views (for now)
- graph: implement gather_params
- graph: implement save_tensors and import_tensors
- tensor: function for setting up tensors (theres a lot of boilerplate)
- tensor: nicer tensor print function
- misc: rename stuff (tgraph -> graph, tt -> tensor, ttuple -> tuple etc)
- tensor: add `track_gradients` to each operation, and remove `tt_destroy_grads`
- tensor: free structs that get copied (like `copy` in `tt_sub`)
- misc: tests + github ci
- misc: move `scce`, `flatten` etc to `functions.c` or something
- misc: don't allcoate memory when doing ops, especially when training.
- graph: make `tgraph_print` look better with bfs instead of dfs

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

- broadcasting, keepdim, proper views, strides, proper storage abstraction (like numpy)
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
