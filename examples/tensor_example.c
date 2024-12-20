#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

// TODO:
// encapsulating ops with functions is broken, i dont think gradients flow
// correcly. check summing with (2, 1, 1) or something with ones get name of
// linspace/arange correct get this working correctly, compare with proper
// tinygrad/pytorch impl variable shapes etc. add to hl_ops or something. need
// to free stuff in function if not being used later. use getenv for batchsize,
// learning_rate, etc other params. add training param to each function

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

tt *mean(tt* input, int axis) {
  int size;
  if (axis == -1) {
    size = ttuple_prod(input->view->shape);
  } else {
    size = input->view->shape->items[axis];
  }
  tt* summed = tt_sum(input, axis);
  tt* div = tt_fill(summed->view->shape, 1.0f / size, true);
  return tt_mul(summed, div);
}

// tt *log_softmax(tt* input, int axis) {
  // tt* exp = tt_exp(input);
  // tt* sum_exp = tt_sum(exp, axis);
  // tt* log_sum_exp = tt_log(sum_exp);
  // tt* expand_log_sum_exp = tt_expand(log_sum_exp, axis, input->view->shape->items[axis]);// would be fixed with broadcasting
  // return tt_sub(input, expand_log_sum_exp);
// }

// torch gives out if out of bounds, tinygrad doesnt.
// we give out.
tt *sparse_categorical_cross_entropy(tt* input, tt* Y) {
  assert(Y->view->shape->size == 1);
  assert(input->view->shape->size == 2);
  assert(Y->view->shape->items[0] == input->view->shape->items[0]);

  tt* one_hot_y = tt_zeros(input->view->shape, false);
  for (int i = 0; i < Y->view->shape->items[0]; i++) {
    int position = (int)Y->data->buffer[i];
    assert(position >= 0 && position < input->view->shape->items[1]);
    one_hot_y->data->buffer[i * input->view->shape->items[1] + position] = 1;
  }

  tt* guesses = tt_mul(input, one_hot_y);
  tt* no_zeros = tt_sum(guesses, 1);

  tt* exp_all = tt_exp(input);
  tt* sum_all = tt_sum(exp_all, 1);
  tt* log_sum_all = tt_log(sum_all);

  tt* sub = tt_sub(log_sum_all, no_zeros);
  return mean(sub, -1);
}

int main(void) {
  srand(time(NULL));

  ttuple* input_shape = ttuple_build(2, 3, 3);
  float input_buffer[9] = {5.0, -5.0, -5.0, 15, 35, 40, -19, 0.001, 2};
  tt* input = tt_from_buffer(input_shape, &input_buffer[0], true);

  ttuple* target_shape = ttuple_build(1, 3);
  float target_buffer[3] = {1, 2, 0};
  tt* target = tt_from_buffer(target_shape, &target_buffer[0], false);

  tt* loss = sparse_categorical_cross_entropy(input, target);

  tgraph* comp_graph = tgraph_build(loss);
  tgraph_zeroed(comp_graph);
  tgraph_backprop(comp_graph);


  tt_print(target, false, false);
  tt_print(input, false, true);
  tt_print(loss, false, true);

}
