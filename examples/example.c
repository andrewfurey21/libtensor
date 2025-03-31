#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <stdint.h>
#include <time.h>

int main(void) {
  srand(time(NULL));
  int batch_size = envvar("BS", 1);

  intarray *input_batch_shape = intarray_build(4, batch_size, 1, 5, 5);
  tensor *input_batch = tensor_linspace(input_batch_shape, 0, 5*5-1, true);

  intarray *conv_shape = intarray_build(4, 1, 1, 3, 3);
  tensor *conv_weights = tensor_linspace(conv_shape, 0, 3*3-1, true);

  tensor* conv = tensor_conv2d(input_batch, conv_weights, true);

  tensor* input_sum = tensor_sum(input_batch,-1, true);

  tensor* conv_sum = tensor_sum(conv, -1, true);

  tensor* output = tensor_add(conv_sum, input_sum, true);

  graph *network = graph_build(output);
  graph_zeroed(network);
  graph_backprop(network);

  tensor_print(input_batch, true, true);
  tensor_print(conv, true, true);
  tensor_print(output, true, true);
}
