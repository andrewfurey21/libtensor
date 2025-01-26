// Convolution examples

#include "../include/tensor.h"

int main(void) {


  // shapes
  ttuple* input_shape = ttuple_build(4, 1, 8, 28, 28);
  ttuple* kernel5x5 = ttuple_build(4, 16, 8, 3, 3);

  // tensors
  tt* input = tt_uniform(input_shape, -1, 1, false);
  tt* kernels = tt_uniform(kernel5x5, -1, 1, false);

  // layers
  tt* output = tt_conv2d(input, kernels);
  // tt_free(input);
  // tt_free(kernels);
  // tt_free(output);
  //
  // // shapes
  // ttuple* kernel3x3 = ttuple_build(4, 1, 1, 3, 3);
  //
  // // tensors
  // input = tt_uniform(input_shape, -1, 1, false);
  // kernels = tt_uniform(kernel3x3, -1, 1, false);
  //
  // // layers
  // output = tt_conv2d(input, kernels);
  // output = tt_conv2d(output, kernels);
  // tt_free(input);
  // tt_free(kernels);
  // tt_free(output);
}
