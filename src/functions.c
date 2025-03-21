#include "assert.h"
#include "../include/tensor.h"

tensor *flatten(tensor *input, int start_dim) {
  assert(start_dim >= 0 && start_dim < input->dview->shape->size);
  intarray *new_shape = intarray_zeros(start_dim + 1);
  uint64_t end = 1;
  for (int i = 0; i < input->dview->shape->size; i++) {
    if (i >= start_dim) {
      end *= input->dview->shape->items[i];
    } else {
      new_shape->items[i] = input->dview->shape->items[i];
    }
  }
  new_shape->items[start_dim] = end;
  tensor *flattened = tensor_reshape(input, new_shape, input->requires_grad);
  return flattened;
}

tensor *mean(tensor *input, int axis) {
  int size;
  if (axis == -1) {
    size = intarray_prod(input->dview->shape);
  } else {
    size = input->dview->shape->items[axis];
  }
  tensor *summed = tensor_sum(input, axis, input->requires_grad);
  tensor *div = tensor_fill(summed->dview->shape, 1.0f / size, true);
  return tensor_mul(summed, div, input->requires_grad);
}

// tensor *variance(tensor *input, int axis, int correction) {
//   tensor *m = mean(input, axis);
//   tensor *expanded_m = tensor_expand(m, axis, input->dview->shape->items[axis]);
//   tensor *sub = tensor_sub(input, expanded_m);
//
//   tensor *sq = tensor_square(sub);
//   tensor *sum = tensor_sum(sq, axis);
//
//   tensor *number =
//       tensor_fill(sum->dview->shape,
//               1.0f / (input->dview->shape->items[axis] - correction), false);
//
//   return tensor_mul(sum, number);
// }

// torch gives out if out of bounds, tinygrad doesnt.
// we give out.
tensor *sparse_categorical_cross_entropy(tensor *input, tensor *Y) {
  assert(Y->dview->shape->size == 1);
  // assert(input->dview->shape->size == 2);
  // assert(Y->dview->shape->items[0] == input->dview->shape->items[0]);

  tensor *one_hot_y = tensor_zeros(input->dview->shape, false);
  for (int i = 0; i < Y->dview->shape->items[0]; i++) {
    int position = (int)Y->data->buffer[i];
    assert(position >= 0 && position < input->dview->shape->items[1]);
    one_hot_y->data->buffer[i * input->dview->shape->items[1] + position] = 1;
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
// TODO: double check gradients work
tensor *log_softmax(tensor *input) {
  tensor *exp_input = tensor_exp(input, input->requires_grad);
  tensor *sum_exp_input = tensor_sum(exp_input, -1, input->requires_grad);
  tensor *log_sum_exp_input = tensor_log(sum_exp_input, input->requires_grad);
  tensor *expanded = tensor_expand(log_sum_exp_input, 0, input->data->size, input->requires_grad);
  intarray *new_shape = intarray_build(1, input->dview->shape->items[1], input->requires_grad);
  tensor *reshaped_input = tensor_reshape(input, new_shape, input->requires_grad);
  intarray_free(new_shape);
  return tensor_sub(reshaped_input, expanded, input->requires_grad);
}

// 2d matmul
// tensor *linear_layer(tensor *input, tensor *weights) {
//   int input_width = input->dview->shape->items[1];
//   int input_height = input->dview->shape->items[0];
//
//   int weights_width = weights->dview->shape->items[1];
//   int weights_height = weights->dview->shape->items[0];
//
//   assert(input_width == weights_height);
//
//   intarray *new_input_shape = intarray_build(3, input_height, input_width, 1);
//   tensor *reshaped_input = tensor_reshape(input, new_input_shape);
//
//   intarray *new_weights_shape = intarray_build(3, 1, weights_height, weights_width);
//   tensor *reshaped_weights = tensor_reshape(weights, new_weights_shape);
//
//   tensor *expanded_input = tensor_expand(reshaped_input, 2, weights_width);
//   tensor *expanded_weights = tensor_expand(reshaped_weights, 0, input_height);
//
//   tensor *mul = tensor_mul(expanded_input, expanded_weights);
//
//   tensor *output = tensor_sum(mul, 1);
//
//   intarray *new_output_shape = intarray_zeros(2);
//   new_output_shape->items[0] = output->dview->shape->items[0];
//   new_output_shape->items[1] = output->dview->shape->items[2];
//
//   tensor *reshaped_output = tensor_reshape(output, new_output_shape);
//   return reshaped_output;
// }
