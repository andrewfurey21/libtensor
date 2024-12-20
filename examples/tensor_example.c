// model based off of https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

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
  assert(input->view->shape->size == 2);
  assert(Y->view->shape->items[0] == input->view->shape->items[0]);

  tt *one_hot_y = tt_zeros(input->view->shape, false);
  for (int i = 0; i < Y->view->shape->items[0]; i++) {
    int position = (int)Y->data->buffer[i];
    assert(position >= 0 && position < input->view->shape->items[1]);
    one_hot_y->data->buffer[i * input->view->shape->items[1] + position] = 1;
  }

  tt *guesses = tt_mul(input, one_hot_y);
  tt *no_zeros = tt_sum(guesses, 1);

  tt *exp_all = tt_exp(input);
  tt *sum_all = tt_sum(exp_all, 1);
  tt *log_sum_all = tt_log(sum_all);

  tt *sub = tt_sub(log_sum_all, no_zeros);
  return mean(sub, -1);
}

// gamma = scale, beta = shift
// like pytorch, expects 4d input, 4d output
// batch axis: 0
tt *batch_norm_2d(tt *input, tt *gamma, tt *beta, tt *running_mean,
                  tt *running_var, float eps, bool update_stats) {
  assert(input->view->shape->size == 4);

  tt *eps_tensor = tt_fill(input->view->shape, eps, false);

  tt *current_mean = mean(input, 0);
  tt *expand_mean =
      tt_expand(current_mean, 0,
                input->view->shape->items[0]); // really need broadcasting
  tt *current_var = variance(input, 0, 1);
  tt *expand_var = tt_expand(current_var, 0, input->view->shape->items[0]);

  tt *input_sub_mean = tt_sub(input, expand_mean);
  tt *plus_eps = tt_add(expand_var, eps_tensor);
  tt *sqrt = tt_sqrt(plus_eps);
  tt *recip = tt_reciprocal(sqrt);
  tt *x_hat = tt_mul(input_sub_mean, recip);

  tt *shift = tt_mul(gamma, x_hat);
  tt *scale = tt_add(beta, shift);

  // need to fix grads within grads bug, cuz memory leaks will be a bitch
  if (update_stats) {
    // update running var and running mean
    tt *original = tt_fill(input->view->shape, 0.99, false);
    tt *current = tt_fill(input->view->shape, 0.01, false);
    tt *original_rmean = tt_mul(running_mean, original);
    tt *current_rmean = tt_mul(expand_mean, current);
    tt_free(running_mean);
    running_mean = tt_add(original_rmean, current_rmean);

    tt *original_rvar = tt_mul(running_var, original);
    tt *current_rvar = tt_mul(expand_var, current);
    tt_free(running_var);
    running_mean = tt_add(original_rvar, current_rvar);

    tt_free(original_rvar);
    tt_free(current_rvar);

    tt_free(original_rmean);
    tt_free(current_rmean);

    tt_free(original);
    tt_free(current);
  }

  return scale;
}

int main(void) {
  srand(time(NULL));

  ttuple *input_shape = ttuple_build(4, 2, 1, 4, 4);
  tt *input = tt_linspace(input_shape, 1, 2 * 1 * 4 * 4, 2 * 1 * 4 * 4, true);

  tt *gamma = tt_ones(input_shape, true);
  tt *beta = tt_zeros(input_shape, true);

  tt *running_mean = tt_zeros(input_shape, true);
  tt *running_var = tt_ones(input_shape, true);

  tt* output = batch_norm_2d(input, gamma, beta, running_mean, running_var, 0.00001, true);
  
  tt *sum = tt_sum(output, -1);

  tgraph *comp_graph = tgraph_build(sum);
  tgraph_zeroed(comp_graph);
  tgraph_backprop(comp_graph);

  tt_print(input, false, true);
  tt_print(output, false, true);
}
