#include "../include/tensor.h"
#include "gtest/gtest.h"
#include "cmath"

TEST(Backwards, Add) {
  intarray *shape = intarray_build(1, 9);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *tensor2 = tensor_linspace(shape, -20, 20, true);

  tensor *output = tensor_add(tensor1, tensor2, true);
  tensor *sum = tensor_sum(output, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], tensor1);
  EXPECT_EQ(output->parents[1], tensor2);
  EXPECT_EQ(output->op, ADD);
  EXPECT_EQ(output->_backwards, _add_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  tensor *correct_grads = tensor_ones(shape, false);

  EXPECT_TRUE(tensor_equal(correct_grads, tensor1->grads, 1e-5, 1e-8))
      << "Backwards add (1) failed\n";
  EXPECT_TRUE(tensor_equal(correct_grads, tensor2->grads, 1e-5, 1e-8))
      << "Backwards add (2) failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads, true, false);
    printf("Tensor 1: \n");
    tensor_print(tensor1->grads, true, false);
    printf("Tensor 2: \n");
    tensor_print(tensor2->grads, true, false);
  }

  graph_backprop(g);

  tensor *correct_grads2 = tensor_fill(shape, 2.0f, false);
  EXPECT_TRUE(tensor_equal(correct_grads2, tensor1->grads, 1e-5, 1e-8))
      << "Backwards add (second backward pass) failed\n";
  EXPECT_TRUE(tensor_equal(correct_grads2, tensor2->grads, 1e-5, 1e-8))
      << "Backwards add (second backward pass) failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads2, true, false);
    printf("Tensor 1: \n");
    tensor_print(tensor1->grads, true, false);
    printf("Tensor 2: \n");
    tensor_print(tensor2->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct_grads);
  graph_free(g);
}

TEST(Backwards, Sub) {
  intarray *shape = intarray_build(1, 9);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *tensor2 = tensor_linspace(shape, -20, 20, true);

  tensor *output = tensor_sub(tensor1, tensor2, true);
  tensor *sum = tensor_sum(output, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], tensor1);
  EXPECT_EQ(output->parents[1], tensor2);
  EXPECT_EQ(output->op, SUB);
  EXPECT_EQ(output->_backwards, _sub_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  tensor *correct_grads1 = tensor_ones(shape, false);
  tensor *correct_grads2 = tensor_neg(correct_grads1, false);

  EXPECT_TRUE(tensor_equal(correct_grads1, tensor1->grads, 1e-5, 1e-8))
      << "Backwards sub lhs failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads1, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  EXPECT_TRUE(tensor_equal(correct_grads2, tensor2->grads, 1e-5, 1e-8))
      << "Backwards sub rhs failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads2, true, false);
    printf("Output: \n");
    tensor_print(tensor2->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct_grads1);
  tensor_free(correct_grads2);
  graph_free(g);
}

TEST(Backwards, Mul) {
  intarray *shape = intarray_build(1, 9);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *tensor2 = tensor_linspace(shape, -20, 20, true);

  tensor *output = tensor_mul(tensor1, tensor2, true);
  tensor *sum = tensor_sum(output, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], tensor1);
  EXPECT_EQ(output->parents[1], tensor2);
  EXPECT_EQ(output->op, MUL);
  EXPECT_EQ(output->_backwards, _mul_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(tensor2, tensor1->grads, 1e-5, 1e-8))
      << "Backwards mul lhs failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(tensor2, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  EXPECT_TRUE(tensor_equal(tensor1, tensor2->grads, 1e-5, 1e-8))
      << "Backwards mul rhs failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(tensor1, true, false);
    printf("Output: \n");
    tensor_print(tensor2->grads, true, false);
  }

  intarray_free(shape);
  graph_free(g);
}

TEST(Backwards, Sum) {
  intarray *shape = intarray_build(2, 4, 4);
  intarray *new_shape = intarray_build(1, 4);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);

  tensor *output1 = tensor_sum(tensor1, 1, true);
  tensor *reshape_output1 = tensor_reshape(output1, new_shape, true);
  tensor *sum = tensor_sum(reshape_output1, 0, true);

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, SUM_REDUCE);
  EXPECT_EQ(output1->_backwards, _sum_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  tensor *correct_grads = tensor_ones(shape, false);
  EXPECT_TRUE(tensor_equal(correct_grads, tensor1->grads, 1e-5, 1e-8))
      << "Backwards sum failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  intarray_free(new_shape);
  tensor_free(correct_grads);
  graph_free(g);
}

TEST(Backwards, Relu) {
  intarray *shape = intarray_build(1, 17);
  float buffer[] = {0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
                    0.000000, 0.000000, 0.000000, 1.000000, 1.000000, 1.000000,
                    1.000000, 1.000000, 1.000000, 1.000000, 1.000000};

  tensor *correct = tensor_from_buffer(shape, buffer, false);
  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);

  tensor *output1 = tensor_relu(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, RELU);
  EXPECT_EQ(output1->_backwards, _relu_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Relu failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  graph_free(g);
}

TEST(Backwards, Reshape) {
  intarray *shape = intarray_build(4, 2, 3, 4, 6);
  intarray *new_shape = intarray_build(1, 144);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_reshape(tensor1, new_shape, true);
  tensor *sum = tensor_sum(output1, 0, true);

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, RESHAPE);
  EXPECT_EQ(output1->_backwards, _reshape_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(intarray_equal(shape, tensor1->grads->vw->shape))
      << "Backwards Reshape failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    intarray_print(shape);
    printf("Output: \n");
    intarray_print(tensor1->grads->vw->shape);
  }

  intarray_free(shape);
  graph_free(g);
}

TEST(Backwards, Expand) {
  intarray *shape = intarray_build(3, 4, 1, 4);
  intarray *final_shape = intarray_build(1, 64);
  int axis = 1;
  int amount = 4;

  tensor *correct = tensor_fill(shape, amount, false);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);

  tensor *output1 = tensor_expand(tensor1, axis, amount, true);
  tensor *reshaped = tensor_reshape(output1, final_shape, true);
  tensor *sum = tensor_sum(reshaped, 0, true);

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, EXPAND);
  EXPECT_EQ(output1->_backwards, _expand_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  tensor *correct_grads = tensor_ones(shape, false);
  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards expand failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_grads, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  intarray_free(final_shape);
  tensor_free(correct_grads);
  graph_free(g);
}

TEST(Backwards, Neg) {
  intarray *shape = intarray_build(1, 17);

  tensor *correct = tensor_fill(shape, -1, false);
  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_neg(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, NEG);
  EXPECT_EQ(output1->_backwards, _neg_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Neg failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  graph_free(g);
  tensor_free(correct);
}

TEST(Backwards, MaxPool2d) {
  intarray *shape1 = intarray_build(4, 1, 1, 4, 4);
  int kernel_size = 2;
  float buffer1[] = {14.00, 16.017, 37.04,   31.95, 26.455, -14.9,
                     -42.1, -7.619, 20.62,   17.15, 15.8,   -46.54,
                     -22.6, 36.5,   -44.103, 10.8};

  intarray *output_shape = intarray_build(4, 1, 1, 2, 2);

  float correct_buffer[] = {
      0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
  };
  tensor *correct = tensor_from_buffer(shape1, correct_buffer, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, true);
  tensor *output = tensor_maxpool2d(tensor1, kernel_size, true);

  intarray *new_shape = intarray_build(1, 4);
  tensor *reshaped = tensor_reshape(output, new_shape, true);
  tensor *sum = tensor_sum(reshaped, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], tensor1);
  EXPECT_EQ(output->op, MAX_POOL_2D);
  EXPECT_EQ(output->_backwards, _maxpool2d_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards max pool 2d failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape1);
  intarray_free(output_shape);
  intarray_free(new_shape);

  graph_free(g);
}

TEST(Backwards, Matmul) {
  intarray *batch_shape = intarray_build(3, 2, 5, 4);
  intarray *weights_shape = intarray_build(2, 3, 5);

  tensor *batch = tensor_linspace(batch_shape, -10, 20, true);
 tensor *weights = tensor_linspace(weights_shape, -10, 20, true);

  float batch_grads_buffer[] = {
      2.1428556, 2.1428556, 2.1428556, 2.1428556, 8.571428,
      8.571428,  8.571428,  8.571428,  14.999998, 14.999998,
      14.999998, 14.999998, 21.42857,  21.42857,  21.42857,
      21.42857,  27.857141, 27.857141, 27.857141, 27.857141,

      2.1428556, 2.1428556, 2.1428556, 2.1428556, 8.571428,
      8.571428,  8.571428,  8.571428,  14.999998, 14.999998,
      14.999998, 14.999998, 21.42857,  21.42857,  21.42857,
      21.42857,  27.857141, 27.857141, 27.857141, 27.857141};

  tensor *batch_grads =
      tensor_from_buffer(batch_shape, batch_grads_buffer, false);

  float weights_grads_buffer[] =
  {-9.230768, 15.384615, 40., 64.61539, 89.230774,
                                  -9.230768, 15.384615, 40., 64.61539, 89.230774,
                                  -9.230768, 15.384615, 40., 64.61539, 89.230774};

  tensor *weights_grads =
      tensor_from_buffer(weights_shape, weights_grads_buffer, false);

  tensor *output = tensor_matmul(weights, batch, true);

  intarray *new_shape = intarray_build(1, 24);
  tensor *reshaped = tensor_reshape(output, new_shape, true);
  tensor *sum = tensor_sum(reshaped, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], weights);
  EXPECT_EQ(output->parents[1], batch);
  EXPECT_EQ(output->op, MATMUL);
  EXPECT_EQ(output->_backwards, _matmul_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(batch_grads, batch->grads, 1e-5, 1e-8))
      << "Backwards matmul failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(batch_grads, true, false);
    printf("Output: \n");
    tensor_print(batch->grads, true, false);
  }

  EXPECT_TRUE(tensor_equal(weights_grads, weights->grads, 1e-5, 1e-8))
      << "Backward matmul failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(weights_grads, true, false);
    printf("Output: \n");
    tensor_print(weights->grads, true, false);
  }

  intarray_free(batch_shape);
  intarray_free(weights_shape);
  intarray_free(new_shape);

  graph_free(g);
  tensor_free(batch_grads);
  tensor_free(weights_grads);
}

TEST(Backwards, Conv2d) {
  intarray *input_shape = intarray_build(4, 1, 2, 5, 5);
  intarray *kernel_shape = intarray_build(4, 2, 2, 3, 3);

  tensor *input = tensor_linspace(input_shape, -10, 15, true);
  tensor *kernels = tensor_linspace(kernel_shape, -10, 15, true);

  float correct_input_grads[] = {
      -7.142857,  -12.857141, -17.142857, -9.999999, -4.285713, -10.000000,
      -17.142853, -21.428566, -11.428569, -4.285713, -8.571428, -12.857141,
      -12.857139, -4.285707,  0.000002,   -1.428572, -0.000001, 4.285716,
      5.714288,   4.285715,   1.428571,   4.285714,  8.571428,  7.142858,
      4.285714,   5.714287,   12.857144,  21.428574, 15.714287, 8.571430,
      15.714287,  34.285717,  55.714287,  40.000004, 21.428574, 30.000002,
      64.285721,  102.857147, 72.857147,  38.571430, 24.285713, 51.428574,
      81.428574,  57.142860,  30.000000,  14.285714, 30.000002, 47.142860,
      32.857147,  17.142857
  };
  tensor *correct_input =
      tensor_from_buffer(input_shape, correct_input_grads, false);

  float correct_kernel_grads[] = {
      -62.448982, -57.857143, -53.265305, -39.489796, -34.897961,
      -30.306122, -16.530611, -11.938776, -7.346940,  52.346939,
      56.938774,  61.530609,  75.306122,  79.897957,  84.489792,
      98.265312,  102.857147, 107.448975, -62.448982, -57.857143,
      -53.265305, -39.489796, -34.897961, -30.306122, -16.530611,
      -11.938776, -7.346940,  52.346939,  56.938774,  61.530609,
      75.306122,  79.897957,  84.489792,  98.265312,  102.857147,
      107.448975

  };
  tensor *correct_kernel =
      tensor_from_buffer(kernel_shape, correct_kernel_grads, false);

  tensor *output = tensor_conv2d(input, kernels, true);

  intarray *final_shape = intarray_build(1, 18);
  tensor *reshaped = tensor_reshape(output, final_shape, true);
  tensor *sum = tensor_sum(reshaped, 0, true);

  EXPECT_TRUE(output->requires_grad);
  EXPECT_EQ(output->parents[0], input);
  EXPECT_EQ(output->parents[1], kernels);
  EXPECT_EQ(output->op, CONV_2D);
  EXPECT_EQ(output->_backwards, _conv2d_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct_input, input->grads, 1e-5, 1e-7))
      << "Backwards conv2d failed for input\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_input, true, false);
    printf("Output: \n");
    tensor_print(input->grads, true, false);
  }

  EXPECT_TRUE(tensor_equal(correct_input, input->grads, 1e-5, 1e-7))
      << "Backwards conv2d failed for weights\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct_kernel, true, false);
    printf("Output: \n");
    tensor_print(kernels->grads, true, false);
  }

  intarray_free(final_shape);
  intarray_free(input_shape);
  intarray_free(kernel_shape);

  tensor_free(correct_input);
  tensor_free(correct_kernel);

  graph_free(g);
}

TEST(Backwards, Square) {
  intarray *shape = intarray_build(1, 17);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_square(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  tensor *correct=  tensor_zeros(tensor1->vw->shape, false);
  for (int i = 0; i < intarray_prod(output1->vw->shape); i++) {
    correct->data->buffer[i] = tensor1->data->buffer[i] * 2;
  }

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, SQUARE);
  EXPECT_EQ(output1->_backwards, _square_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Square failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct);
  graph_free(g);
}

TEST(Backwards, Sqrt) {
  intarray *shape = intarray_build(1, 17);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_sqrt(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  tensor *correct=  tensor_zeros(tensor1->vw->shape, false);
  for (int i = 0; i < intarray_prod(output1->vw->shape); i++) {
    correct->data->buffer[i] = 1.0f / (sqrtf(tensor1->data->buffer[i]) * 2.0f);
  }

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, SQRT);
  EXPECT_EQ(output1->_backwards, _sqrt_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Sqrt failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct);
  graph_free(g);
}

TEST(Backwards, Exp) {
  intarray *shape = intarray_build(1, 17);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_exp(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  tensor *correct=  tensor_zeros(tensor1->vw->shape, false);
  for (int i = 0; i < intarray_prod(output1->vw->shape); i++) {
    correct->data->buffer[i] = exp(tensor1->data->buffer[i]);
  }

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, EXP);
  EXPECT_EQ(output1->_backwards, _exp_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Exp failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct);
  graph_free(g);
}

TEST(Backwards, Log) {
  intarray *shape = intarray_build(1, 17);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_log(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  tensor *correct=  tensor_zeros(tensor1->vw->shape, false);
  for (int i = 0; i < intarray_prod(output1->vw->shape); i++) {
    correct->data->buffer[i] = 1.0f / tensor1->data->buffer[i];
  }

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, LOG);
  EXPECT_EQ(output1->_backwards, _log_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Log failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct);
  graph_free(g);
}

TEST(Backwards, Reciprocal) {
  intarray *shape = intarray_build(1, 17);

  tensor *tensor1 = tensor_linspace(shape, -10, 10, true);
  tensor *output1 = tensor_reciprocal(tensor1, true);
  tensor *sum = tensor_sum(output1, 0, true);

  tensor *correct=  tensor_zeros(tensor1->vw->shape, false);
  for (int i = 0; i < intarray_prod(output1->vw->shape); i++) {
    float val = tensor1->data->buffer[i];
    correct->data->buffer[i] = -1.0f / (val * val);
  }

  EXPECT_TRUE(output1->requires_grad);
  EXPECT_EQ(output1->parents[0], tensor1);
  EXPECT_EQ(output1->op, RECIPROCAL);
  EXPECT_EQ(output1->_backwards, _reciprocal_backwards);

  graph *g = graph_build(sum);
  graph_zeroed(g);
  graph_backprop(g);

  EXPECT_TRUE(tensor_equal(correct, tensor1->grads, 1e-5, 1e-8))
      << "Backwards Reciprocal failed\n";

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(tensor1->grads, true, false);
  }

  intarray_free(shape);
  tensor_free(correct);
  graph_free(g);
}
