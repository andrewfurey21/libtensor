#include "assert.h"
#include "../include/tensor.h"

tt *flatten(tt *input, int start_dim) {
  assert(start_dim >= 0 && start_dim < input->view->shape->size);
  ttuple *new_shape = ttuple_zeros(start_dim + 1);
  uint64_t end = 1;
  for (int i = 0; i < input->view->shape->size; i++) {
    if (i >= start_dim) {
      end *= input->view->shape->items[i];
    } else {
      new_shape->items[i] = input->view->shape->items[i];
    }
  }
  new_shape->items[start_dim] = end;
  tt *flattened = tt_reshape(input, new_shape);
  return flattened;
}

tt *mean(tt *input, int axis) {
  int size;
  if (axis == -1) {
    size = ttuple_prod(input->view->shape);
  } else {
    size = input->view->shape->items[axis];
  }
  tt *summed = tt_sum(input, axis);
  tt *div = tt_fill(summed->view->shape, 1.0f / size, true);
  return tt_mul(summed, div);
}

tt *variance(tt *input, int axis, int correction) {
  tt *m = mean(input, axis);
  tt *expanded_m = tt_expand(m, axis, input->view->shape->items[axis]);
  tt *sub = tt_sub(input, expanded_m);

  tt *sq = tt_square(sub);
  tt *sum = tt_sum(sq, axis);

  tt *number =
      tt_fill(sum->view->shape,
              1.0f / (input->view->shape->items[axis] - correction), false);

  return tt_mul(sum, number);
}

// torch gives out if out of bounds, tinygrad doesnt.
// we give out.
tt *sparse_categorical_cross_entropy(tt *input, tt *Y) {
  assert(Y->view->shape->size == 1);
  // assert(input->view->shape->size == 2);
  // assert(Y->view->shape->items[0] == input->view->shape->items[0]);

  tt *one_hot_y = tt_zeros(input->view->shape, false);
  for (int i = 0; i < Y->view->shape->items[0]; i++) {
    int position = (int)Y->data->buffer[i];
    assert(position >= 0 && position < input->view->shape->items[1]);
    one_hot_y->data->buffer[i * input->view->shape->items[1] + position] = 1;
  }

  tt *guesses = tt_mul(input, one_hot_y);
  tt *no_zeros = tt_sum(guesses, -1);

  tt *exp_all = tt_exp(input);
  tt *sum_all = tt_sum(exp_all, -1);
  tt *log_sum_all = tt_log(sum_all);

  tt *sub = tt_sub(log_sum_all, no_zeros);
  return mean(sub, -1);
}

// takes a vector! (1, n)
// TODO: double check gradients work
tt *log_softmax(tt *input) {
  tt *exp_input = tt_exp(input);
  tt *sum_exp_input = tt_sum(exp_input, -1);
  tt *log_sum_exp_input = tt_log(sum_exp_input);
  tt *expanded = tt_expand(log_sum_exp_input, 0, input->data->size);
  ttuple *new_shape = ttuple_build(1, input->view->shape->items[1]);
  tt *reshaped_input = tt_reshape(input, new_shape);
  ttuple_free(new_shape);
  return tt_sub(reshaped_input, expanded);
}

// 2d matmul
tt *linear_layer(tt *input, tt *weights) {
  int input_width = input->view->shape->items[1];
  int input_height = input->view->shape->items[0];

  int weights_width = weights->view->shape->items[1];
  int weights_height = weights->view->shape->items[0];

  assert(input_width == weights_height);

  ttuple *new_input_shape = ttuple_build(3, input_height, input_width, 1);
  tt *reshaped_input = tt_reshape(input, new_input_shape);

  ttuple *new_weights_shape = ttuple_build(3, 1, weights_height, weights_width);
  tt *reshaped_weights = tt_reshape(weights, new_weights_shape);

  tt *expanded_input = tt_expand(reshaped_input, 2, weights_width);
  tt *expanded_weights = tt_expand(reshaped_weights, 0, input_height);

  tt *mul = tt_mul(expanded_input, expanded_weights);

  tt *output = tt_sum(mul, 1);

  ttuple *new_output_shape = ttuple_zeros(2);
  new_output_shape->items[0] = output->view->shape->items[0];
  new_output_shape->items[1] = output->view->shape->items[2];

  tt *reshaped_output = tt_reshape(output, new_output_shape);
  return reshaped_output;
}
