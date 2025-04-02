#include "../include/tensor.h"
#include "assert.h"
#include "math.h"

tensor *flatten(tensor *input, int start_dim) {
  assert(start_dim >= 0 && start_dim < input->vw->shape->size);
  intarray *new_shape = intarray_zeros(start_dim + 1);
  uint64_t end = 1;
  for (int i = 0; i < input->vw->shape->size; i++) {
    if (i >= start_dim) {
      end *= input->vw->shape->items[i];
    } else {
      new_shape->items[i] = input->vw->shape->items[i];
    }
  }
  new_shape->items[start_dim] = end;
  tensor *flattened = tensor_reshape(input, new_shape, input->requires_grad);
  intarray_free(new_shape);
  return flattened;
}

tensor *mean(tensor *input, int axis) {
  int size;
  if (axis == -1) {
    size = intarray_prod(input->vw->shape);
  } else {
    size = input->vw->shape->items[axis];
  }
  tensor *summed = tensor_sum(input, axis, input->requires_grad);
  tensor *div = tensor_fill(summed->vw->shape, 1.0f / size, false);
  return tensor_mul(summed, div, input->requires_grad);
}

tensor *variance(tensor *input, int axis, int correction) {
  tensor *m = mean(input, axis);
  tensor *expanded_m = tensor_expand(m, axis, input->vw->shape->items[axis],
                                     input->requires_grad);
  tensor *sub = tensor_sub(input, expanded_m, input->requires_grad);

  tensor *sq = tensor_square(sub, input->requires_grad);
  tensor *sum = tensor_sum(sq, axis, input->requires_grad);

  tensor *number =
      tensor_fill(sum->vw->shape,
                  1.0f / (input->vw->shape->items[axis] - correction), false);

  return tensor_mul(sum, number, input->requires_grad);
}

tensor *one_hot_encode(tensor *batch, int size) {
  int batch_size = batch->vw->shape->items[0];
  intarray *shape = intarray_build(2, batch_size, size);
  tensor *one_hot_y = tensor_zeros(shape, false);
  for (int i = 0; i < batch_size; i++) {
    int position = storage_getitem(batch->data, i);
    assert(position >= 0 && position < size);
    int index = i * size + position;
    storage_setitem(one_hot_y->data, index, 1);
  }
  return one_hot_y;
}

tensor *cross_entropy(tensor *log_probs, tensor *Y) {
  // tensor* log_probs = log_softmax(input);
  tensor *mul_true = tensor_mul(log_probs, Y, log_probs->requires_grad);
  tensor *sum = tensor_sum(mul_true, 1, log_probs->requires_grad);
  tensor *reduction = mean(sum, -1);
  return tensor_neg(reduction, log_probs->requires_grad);
}

tensor *_max(tensor *input) {
  float max_value = -INFINITY;
  for (int i = 0; i < input->data->size; i++) {
    float current = storage_getitem(input->data, i);
    if (max_value < current) {
      max_value = current;
    }
  }
  return tensor_fill(input->vw->shape, max_value, false);
}

// takes a matrix (n, m)
// softmax(x) = softmax(x+c) so cool! now numerically stable
tensor *log_softmax(tensor *input) {
  tensor *max = _max(input);
  tensor *subbed = tensor_sub(input, max, input->requires_grad);
  tensor *exp_input = tensor_exp(subbed, input->requires_grad);
  tensor *sum_exp_input = tensor_sum(exp_input, 1, input->requires_grad);

  // tensor *epsilon = tensor_fill(sum_exp_input->vw->shape, 1e-12, false);
  // tensor *safe_to_log =
  //     tensor_add(epsilon, sum_exp_input, input->requires_grad);
  // tensor *log_sum_exp_input = tensor_log(safe_to_log, input->requires_grad);
  tensor *log_sum_exp_input = tensor_log(sum_exp_input, input->requires_grad);

  int to_expand = input->vw->shape->items[1];
  tensor *expanded =
      tensor_expand(log_sum_exp_input, 1, to_expand, input->requires_grad);
  return tensor_sub(subbed, expanded, input->requires_grad);
}
