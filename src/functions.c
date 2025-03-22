#include "../include/tensor.h"
#include "assert.h"

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
  tensor *div = tensor_fill(summed->vw->shape, 1.0f / size, true);
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

// torch gives out if out of bounds, tinygrad doesnt.
// we give out.
tensor *sparse_categorical_cross_entropy(tensor *input, tensor *Y) {
  assert(Y->vw->shape->size == 1);

  tensor *one_hot_y = tensor_zeros(input->vw->shape, false);
  for (int i = 0; i < Y->vw->shape->items[0]; i++) {
    int position = storage_getitem(Y->data, i);
    assert(position >= 0 && position < input->vw->shape->items[1]);
    int index = i * input->vw->shape->items[1] + position;
    storage_setitem(one_hot_y->data, index, 1);
  }

  tensor *guesses = tensor_mul(input, one_hot_y, input->requires_grad);
  tensor *no_zeros = tensor_sum(guesses, -1, input->requires_grad);

  tensor *exp_all = tensor_exp(input, input->requires_grad);
  tensor *sum_all = tensor_sum(exp_all, -1, input->requires_grad);
  tensor *log_sum_all = tensor_log(sum_all, input->requires_grad);

  tensor *sub = tensor_sub(log_sum_all, no_zeros, input->requires_grad);
  return mean(sub, -1);
}

// takes a vector! (1, n)
tensor *log_softmax(tensor *input) {
  tensor *exp_input = tensor_exp(input, input->requires_grad);
  tensor *sum_exp_input = tensor_sum(exp_input, -1, input->requires_grad);
  tensor *log_sum_exp_input = tensor_log(sum_exp_input, input->requires_grad);
  tensor *expanded = tensor_expand(log_sum_exp_input, 0, input->data->size,
                                   input->requires_grad);
  intarray *new_shape =
      intarray_build(1, input->vw->shape->items[1], input->requires_grad);
  tensor *reshaped_input =
      tensor_reshape(input, new_shape, input->requires_grad);
  intarray_free(new_shape);
  return tensor_sub(reshaped_input, expanded, input->requires_grad);
}
