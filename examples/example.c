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
