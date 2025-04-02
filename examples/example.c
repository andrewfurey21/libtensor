#include "../include/tensor.h"

int main(void) {
  intarray* tshape = intarray_build(2, 2, 3);
  float buffer[] = {-1, 2, -3, 1, -2, 3};

  intarray* yshape = intarray_build(2, 2, 3);
  float other[] = {0, 1, 0, 0, 0, 1};

  tensor* t = tensor_from_buffer(tshape, buffer, true);
  tensor* y = tensor_from_buffer(yshape, other, false);

  tensor* loss = cross_entropy(t, y);
  tensor_print(loss, true, false);

  // graph* network = graph_build(loss);
  // graph_zeroed(network);
  // graph_backprop(network);
  //
  // tensor_print(loss, true, false);
  // tensor_print(log_probs, false, true);
}
