#include "../include/tensor.h"
#include "gtest/gtest.h"
#include <cmath>

TEST(Tensor, Zeros) {
  intarray *shape = intarray_build(4, 3, 2, 1, 1);
  tensor *zeros = tensor_zeros(shape, false);

  ASSERT_TRUE(intarray_equal(shape, zeros->vw->shape));
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(zeros->data->buffer[i], 0) << "Found not zero";
  }
  EXPECT_FALSE(zeros->grads);
  EXPECT_FALSE(zeros->requires_grad);
  EXPECT_FALSE(zeros->parents);
  EXPECT_EQ(zeros->op, NOOP);
  EXPECT_FALSE(zeros->_backwards);

  intarray_free(shape);
  tensor_free(zeros);
}

TEST(Tensor, Ones) {
  intarray *shape = intarray_build(3, 2, 2, 2);
  tensor *ones = tensor_ones(shape, false);

  ASSERT_TRUE(intarray_equal(shape, ones->vw->shape));
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(ones->data->buffer[i], 1) << "Found not one";
  }

  EXPECT_FALSE(ones->grads);
  EXPECT_FALSE(ones->requires_grad);
  EXPECT_FALSE(ones->parents);
  EXPECT_EQ(ones->op, NOOP);
  EXPECT_FALSE(ones->_backwards);

  intarray_free(shape);
  tensor_free(ones);
}

TEST(Tensor, FromBuffer) {
  intarray *shape = intarray_build(3, 2, 1, 4);
  float buffer[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  tensor *t = tensor_from_buffer(shape, buffer, false);

  ASSERT_TRUE(intarray_equal(shape, t->vw->shape));
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(t->data->buffer[i], buffer[i]) << "Found unequal";
  }

  EXPECT_FALSE(t->grads);
  EXPECT_FALSE(t->requires_grad);
  EXPECT_FALSE(t->parents);
  EXPECT_EQ(t->op, NOOP);
  EXPECT_FALSE(t->_backwards);

  intarray_free(shape);
  tensor_free(t);
}

TEST(Tensor, GetIndex) {
  intarray *shape = intarray_build(4, 6, 5, 4, 4);
  intarray *index = intarray_build(4, 0, 1, 2, 3);
  tensor *t = tensor_linspace(shape, 0, 6 * 5 * 4 * 4 - 1, false);

  EXPECT_EQ(tensor_getindex(t, index), 27);

  intarray_free(shape);
  intarray_free(index);
  tensor_free(t);
}

TEST(Tensor, SetIndex) {
  intarray *shape = intarray_build(3, 2, 3, 4);
  intarray *index = intarray_build(3, 1, 2, 3);
  tensor *t = tensor_zeros(shape, false);

  tensor_setindex(t, index, 27);
  EXPECT_EQ(tensor_getindex(t, index), 27);

  intarray_free(shape);
  intarray_free(index);
  tensor_free(t);
}

TEST(Tensor, Fill) {
  float value = 10.5f;
  intarray *shape = intarray_build(3, 3, 3, 3);
  tensor *t = tensor_fill(shape, value, false);

  ASSERT_TRUE(intarray_equal(shape, t->vw->shape));
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(t->data->buffer[i], value);
  }

  EXPECT_FALSE(t->grads);
  EXPECT_FALSE(t->requires_grad);
  EXPECT_FALSE(t->parents);
  EXPECT_EQ(t->op, NOOP);
  EXPECT_FALSE(t->_backwards);

  intarray_free(shape);
  tensor_free(t);
}

TEST(Tensor, Linspace) {
  intarray *shape = intarray_build(3, 1, 3, 5);
  tensor *t = tensor_linspace(shape, 0, 20, false);

  float correct_buffer[] = {0.0,
                            1.4285714285714286,
                            2.857142857142857,
                            4.285714285714286,
                            5.714285714285714,
                            7.142857142857143,
                            8.571428571428571,
                            10.0,
                            11.428571428571429,
                            12.857142857142858,
                            14.285714285714286,
                            15.714285714285715,
                            17.142857142857142,
                            18.571428571428573,
                            20.0};

  tensor *correct = tensor_from_buffer(shape, correct_buffer, false);
  ASSERT_TRUE(intarray_equal(shape, t->vw->shape));
  EXPECT_TRUE(tensor_equal(correct, t, 1e-5, 1e-8));

  EXPECT_FALSE(t->grads);
  EXPECT_FALSE(t->requires_grad);
  EXPECT_FALSE(t->parents);
  EXPECT_EQ(t->op, NOOP);
  EXPECT_FALSE(t->_backwards);

  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(t, true, false);
  }

  intarray_free(shape);
  tensor_free(t);
  tensor_free(correct);
}

TEST(Tensor, Copy) {
  intarray *shape = intarray_build(3, 3, 3, 3);
  tensor *t = tensor_ones(shape, false);
  tensor *s = tensor_copy(t, false);

  ASSERT_TRUE(intarray_equal(s->vw->shape, t->vw->shape));
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(s->data->buffer[i], 1);
  }

  EXPECT_FALSE(t->grads);
  EXPECT_FALSE(t->requires_grad);
  EXPECT_FALSE(t->parents);
  EXPECT_EQ(t->op, NOOP);
  EXPECT_FALSE(t->_backwards);

  intarray_free(shape);
  tensor_free(t);
  tensor_free(s);
}

TEST(Tensor, ToZeros) {
  intarray *shape = intarray_build(3, 3, 3, 3);
  tensor *t = tensor_ones(shape, false);

  tensor_to_zeros(t);
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(t->data->buffer[i], 0);
  }

  EXPECT_FALSE(t->grads);
  EXPECT_FALSE(t->requires_grad);
  EXPECT_FALSE(t->parents);
  EXPECT_EQ(t->op, NOOP);
  EXPECT_FALSE(t->_backwards);

  intarray_free(shape);
  tensor_free(t);
}

TEST(Tensor, ToN) {
  float value = -19.5;
  intarray *shape = intarray_build(3, 3, 5, 3);
  tensor *t = tensor_ones(shape, false);

  tensor_to_n(t, value);
  for (int i = 0; i < intarray_prod(shape); i++) {
    EXPECT_EQ(t->data->buffer[i], value);
  }

  intarray_free(shape);
  tensor_free(t);
}

TEST(Tensor, Equal) {
  intarray *shape = intarray_build(3, 3, 5, 3);
  tensor *t = tensor_linspace(shape, 0, 10, false);
  tensor *s = tensor_linspace(shape, 0, 10, false);

  EXPECT_TRUE(tensor_equal(t, s, 1e-5, 1e-8));

  intarray_free(shape);
  tensor_free(t);
}

TEST(Tensor, Add3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float buffer2[] = {
      -5.5f, 3.1f,  9.3f,   -0.3f, -3.6f, 7.9f,
      2.4f,  9.19f, -3.18f, 1.6f,  -80.f, -1.0f,
  };

  float correct_output[] = {
      -4.0f, 1.1f,   39.8f,   4.0f, -54.2f, 14.8f,
      10.2f, 17.30f, -12.20f, 0.0f, -100.f, 32.09f,
  };
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_add(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);
  EXPECT_TRUE(equal) << "Elementwise add failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, ADD);
  EXPECT_EQ(output->_backwards, _add_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Add5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  float buffer1[] = {14.002150434963262, 16.017431897728663, 37.04691907670404,
                     31.958888165236004, 26.455945606000817, 14.9314038112844,
                     42.13679912635356,  7.619203380668943,  20.621661190022696,
                     17.15608743014474,  15.828807868019712, 46.54467295499289,
                     22.680455007875118, 36.53114891124966,  44.1033824633299,
                     10.89673409143278,  5.601692850848466,  3.2401870000765722,
                     17.618641915494504, 18.959840953997343, 42.59000522099807,
                     32.44600746645413,  49.34399543891468,  5.800127811424527,
                     27.127713792926166};

  float buffer2[] = {35.64673308181586,  25.242851708470106, 30.86741346043932,
                     0.6428431971494342, 31.47788458119633,  37.61251871576386,
                     3.326919537068612,  14.224674363044649, 1.6449239775347402,
                     30.590547407056707, 23.793698387704794, 16.90710935390682,
                     23.782156012574855, 29.48355591287415,  17.047737285967358,
                     14.192452576725817, 45.6360212433912,   20.160049694188647,
                     14.826107826748524, 32.61610496057655,  4.0797422115779085,
                     11.905514510619476, 38.194205028063195, 15.762772416929227,
                     18.395900010993866};

  float correct_output[] = {
      49.648884, 41.260284, 67.914337, 32.601730, 57.933830,
      52.543922, 45.463718, 21.843878, 22.266586, 47.746635,
      39.622505, 63.451782, 46.462608, 66.014709, 61.151119,
      25.089188, 51.237713, 23.400236, 32.444752, 51.575943,
      46.669746, 44.351521, 87.538200, 21.562901, 45.523613};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_add(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise add failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, ADD);
  EXPECT_EQ(output->_backwards, _add_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Sub3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float buffer2[] = {
      -5.5f, 3.1f,  9.3f,   -0.3f, -3.6f, 7.9f,
      2.4f,  9.19f, -3.18f, 1.6f,  -80.f, -1.0f,
  };

  float correct_output[] = {7.000000,   -5.100000, 21.200001, 4.600000,
                            -47.000000, -1.000000, 5.400000,  -1.080000,
                            -5.840000,  -3.200000, 60.000000, 34.090000};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_sub(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise sub failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SUB);
  EXPECT_EQ(output->_backwards, _sub_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Sub5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  float buffer1[] = {14.002150434963262, 16.017431897728663, 37.04691907670404,
                     31.958888165236004, 26.455945606000817, 14.9314038112844,
                     42.13679912635356,  7.619203380668943,  20.621661190022696,
                     17.15608743014474,  15.828807868019712, 46.54467295499289,
                     22.680455007875118, 36.53114891124966,  44.1033824633299,
                     10.89673409143278,  5.601692850848466,  3.2401870000765722,
                     17.618641915494504, 18.959840953997343, 42.59000522099807,
                     32.44600746645413,  49.34399543891468,  5.800127811424527,
                     27.127713792926166};

  float buffer2[] = {35.64673308181586,  25.242851708470106, 30.86741346043932,
                     0.6428431971494342, 31.47788458119633,  37.61251871576386,
                     3.326919537068612,  14.224674363044649, 1.6449239775347402,
                     30.590547407056707, 23.793698387704794, 16.90710935390682,
                     23.782156012574855, 29.48355591287415,  17.047737285967358,
                     14.192452576725817, 45.6360212433912,   20.160049694188647,
                     14.826107826748524, 32.61610496057655,  4.0797422115779085,
                     11.905514510619476, 38.194205028063195, 15.762772416929227,
                     18.395900010993866};

  float correct_output[] = {
      -21.644581, -9.225420,  6.179508,   31.316046, -5.021938,
      -22.681114, 38.809879,  -6.605471,  18.976738, -13.434460,
      -7.964891,  29.637564,  -1.101702,  7.047594,  27.055645,
      -3.295718,  -40.034328, -16.919863, 2.792535,  -13.656263,
      38.510262,  20.540493,  11.149788,  -9.962645, 8.731813};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_sub(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise sub failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SUB);
  EXPECT_EQ(output->_backwards, _sub_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Mul3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float buffer2[] = {
      -5.5f, 3.1f,  9.3f,   -0.3f, -3.6f, 7.9f,
      2.4f,  9.19f, -3.18f, 1.6f,  -80.f, -1.0f,
  };

  float correct_output[] = {-8.250000,  -6.200000, 283.649994,  -1.290000,
                            182.159988, 54.510002, 18.720001,   74.530891,
                            28.683601,  -2.560000, 1600.000000, -33.090000};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_mul(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise mul failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, MUL);
  EXPECT_EQ(output->_backwards, _mul_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Mul5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  float buffer1[] = {14.002150434963262, 16.017431897728663, 37.04691907670404,
                     31.958888165236004, 26.455945606000817, 14.9314038112844,
                     42.13679912635356,  7.619203380668943,  20.621661190022696,
                     17.15608743014474,  15.828807868019712, 46.54467295499289,
                     22.680455007875118, 36.53114891124966,  44.1033824633299,
                     10.89673409143278,  5.601692850848466,  3.2401870000765722,
                     17.618641915494504, 18.959840953997343, 42.59000522099807,
                     32.44600746645413,  49.34399543891468,  5.800127811424527,
                     27.127713792926166};

  float buffer2[] = {35.64673308181586,  25.242851708470106, 30.86741346043932,
                     0.6428431971494342, 31.47788458119633,  37.61251871576386,
                     3.326919537068612,  14.224674363044649, 1.6449239775347402,
                     30.590547407056707, 23.793698387704794, 16.90710935390682,
                     23.782156012574855, 29.48355591287415,  17.047737285967358,
                     14.192452576725817, 45.6360212433912,   20.160049694188647,
                     14.826107826748524, 32.61610496057655,  4.0797422115779085,
                     11.905514510619476, 38.194205028063195, 15.762772416929227,
                     18.395900010993866};

  float correct_output[] = {
      499.130920, 404.325623, 1143.542603, 20.544554,   832.777222,
      561.607727, 140.185745, 108.380692,  33.921066,   524.814148,
      376.625885, 786.935913, 539.390076,  1077.068237, 751.862854,
      154.651382, 255.638962, 65.322327,   261.215912,  618.396118,
      173.756241, 386.286407, 1884.654663, 91.426102,   499.038727};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape1, buffer2, false);

  tensor *output = tensor_mul(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor1->vw->shape));
  ASSERT_TRUE(intarray_equal(output->vw->shape, tensor2->vw->shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise mul failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, MUL);
  EXPECT_EQ(output->_backwards, _mul_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Sum3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  intarray *output_shape = intarray_build(2, 3, 1);
  int axis = 1;
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float correct_output[] = {34.299999, -27.789997, 2.469999};
  tensor *correct = tensor_from_buffer(output_shape, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_sum(tensor1, axis, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, output_shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Sum reduce failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SUM_REDUCE);
  EXPECT_EQ(output->_backwards, _sum_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Sum5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  int axis = 0;
  intarray *output_shape = intarray_build(2, 1, 5);
  float buffer1[] = {14.002150434963262, 16.017431897728663, 37.04691907670404,
                     31.958888165236004, 26.455945606000817, 14.9314038112844,
                     42.13679912635356,  7.619203380668943,  20.621661190022696,
                     17.15608743014474,  15.828807868019712, 46.54467295499289,
                     22.680455007875118, 36.53114891124966,  44.1033824633299,
                     10.89673409143278,  5.601692850848466,  3.2401870000765722,
                     17.618641915494504, 18.959840953997343, 42.59000522099807,
                     32.44600746645413,  49.34399543891468,  5.800127811424527,
                     27.127713792926166};

  float correct_output[] = {
      98.249100, 142.746597, 119.930756, 112.530472, 133.802963,
  };
  tensor *correct = tensor_from_buffer(output_shape, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_sum(tensor1, axis, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, output_shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Sum reduce failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SUM_REDUCE);
  EXPECT_EQ(output->_backwards, _sum_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Relu3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float correct_output[] = {1.500000,  -0.000000, 30.500000, 4.300000,
                            -0.000000, 6.900000,  7.800000,  8.110000,
                            -0.000000, -0.000000, -0.000000, 33.090000};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_relu(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));
  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Relu failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, RELU);
  EXPECT_EQ(output->_backwards, _relu_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Relu5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  float buffer1[] = {
      14.002150434963262,  16.017431897728663,  37.04691907670404,
      31.958888165236004,  26.455945606000817,  -14.9314038112844,
      -42.13679912635356,  -7.619203380668943,  20.621661190022696,
      17.15608743014474,   15.828807868019712,  -46.54467295499289,
      -22.680455007875118, 36.53114891124966,   -44.1033824633299,
      10.89673409143278,   5.601692850848466,   3.2401870000765722,
      -17.618641915494504, -18.959840953997343, -42.59000522099807,
      32.44600746645413,   49.34399543891468,   5.800127811424527,
      -27.127713792926166};

  float correct_output[] = {
      14.002151, 16.017431, 37.046921, 31.958889, 26.455946,
      -0.000000, -0.000000, -0.000000, 20.621662, 17.156088,
      15.828808, -0.000000, -0.000000, 36.531151, -0.000000,
      10.896734, 5.601693,  3.240187,  -0.000000, -0.000000,
      -0.000000, 32.446007, 49.343994, 5.800128,  -0.000000};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *output = tensor_relu(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Relu failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, RELU);
  EXPECT_EQ(output->_backwards, _relu_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Reshape3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  intarray *new_shape = intarray_build(4, 3, 2, 1, 2);

  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float correct_output[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };
  tensor *correct = tensor_from_buffer(new_shape, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *output = tensor_reshape(tensor1, new_shape, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, new_shape));
  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Reshape failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, RESHAPE);
  EXPECT_EQ(output->_backwards, _reshape_backwards);

  intarray_free(shape1);
  intarray_free(new_shape);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Expand) {
  intarray *shape1 = intarray_build(4, 3, 2, 1, 2);
  intarray *new_shape = intarray_build(4, 3, 2, 5, 2);
  int axis = 2;
  int amount = 5;
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float correct_output[] = {
      1.500000,   -2.000000, 1.500000,   -2.000000, 1.500000,   -2.000000,
      1.500000,   -2.000000, 1.500000,   -2.000000, 30.500000,  4.300000,
      30.500000,  4.300000,  30.500000,  4.300000,  30.500000,  4.300000,
      30.500000,  4.300000,  -50.599998, 6.900000,  -50.599998, 6.900000,
      -50.599998, 6.900000,  -50.599998, 6.900000,  -50.599998, 6.900000,
      7.800000,   8.110000,  7.800000,   8.110000,  7.800000,   8.110000,
      7.800000,   8.110000,  7.800000,   8.110000,  -9.020000,  -1.600000,
      -9.020000,  -1.600000, -9.020000,  -1.600000, -9.020000,  -1.600000,
      -9.020000,  -1.600000, -20.000000, 33.090000, -20.000000, 33.090000,
      -20.000000, 33.090000, -20.000000, 33.090000, -20.000000, 33.090000,
  };
  tensor *correct = tensor_from_buffer(new_shape, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *output = tensor_expand(tensor1, axis, amount, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, new_shape));
  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Expand failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, EXPAND);
  EXPECT_EQ(output->_backwards, _expand_backwards);

  intarray_free(shape1);
  intarray_free(new_shape);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Neg3x4) {
  intarray *shape1 = intarray_build(2, 3, 4);
  float buffer1[] = {
      1.5f, -2.0f, 30.5f,  4.3f,  -50.6f, 6.9f,
      7.8f, 8.11f, -9.02f, -1.6f, -20.f,  33.09f,
  };

  float correct_output[] = {
      -1.5f, 2.0f,   -30.5f, -4.3f, 50.6f, -6.9f,
      -7.8f, -8.11f, 9.02f,  1.6f,  20.f,  -33.09f,
  };

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);

  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_neg(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));
  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Relu failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, NEG);
  EXPECT_EQ(output->_backwards, _neg_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Neg5x5) {
  intarray *shape1 = intarray_build(2, 5, 5);
  float buffer1[] = {
      14.00,  16.0163,  37.04,  31.95, 26.4517, -14.4, -42.1, -7.6143, 20.696,
      17.15,  15.8212,  -46.89, -22.6, 36.531,  -44.1, 10.89, 5.6016,  3.22,
      -17.64, -18.9343, -42.07, 32.44, 49.348,  5.807, -27.1};

  float correct_output[] = {-14.00,   -16.0163, -37.04,  -31.95,  -26.4517,
                            14.4,     42.1,     7.6143,  -20.696, -17.15,
                            -15.8212, 46.89,    22.6,    -36.531, 44.1,
                            -10.89,   -5.6016,  -3.22,   17.64,   18.9343,
                            42.07,    -32.44,   -49.348, -5.807,  27.1};
  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *output = tensor_neg(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Relu failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, NEG);
  EXPECT_EQ(output->_backwards, _neg_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, MaxPool2d5x5) {
  intarray *shape1 = intarray_build(4, 1, 1, 4, 4);
  int kernel_size = 2;
  float buffer1[] = {14.00, 16.017, 37.04,   31.95, 26.455, -14.9,
                     -42.1, -7.619, 20.62,   17.15, 15.8,   -46.54,
                     -22.6, 36.5,   -44.103, 10.8};

  intarray *output_shape = intarray_build(4, 1, 1, 2, 2);
  float correct_output[] = {
      26.455000,
      37.040001,
      36.500000,
      15.800000,
  };
  tensor *correct = tensor_from_buffer(output_shape, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *output = tensor_maxpool2d(tensor1, kernel_size, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, output_shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "max pool 2d failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, MAX_POOL_2D);
  EXPECT_EQ(output->_backwards, _maxpool2d_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Matmul) {
  intarray *shape1 = intarray_build(2, 2, 5);
  float buffer1[] = {14.00, 16.0,  37.04, 31.95, 26.4,
                     -14.9, -42.1, -7.6,  20.62, 17.15};

  intarray *shape2 = intarray_build(3, 1, 5, 3);
  float buffer2[] = {14.02, 16.03, 37.04, 31.94, 26.47,  -14.9, -42.6, -7.63,
                     20.62, 17.1,  15.82, -46.5, -22.18, 36.5,  -44.1};

  intarray *output_shape = intarray_build(3, 1, 2, 3);
  float correct_output[] = {
      -909.790894,  1834.373779, -1605.990234,
      -1257.596802, -343.062622, -1796.463013,
  };

  tensor *correct = tensor_from_buffer(output_shape, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape2, buffer2, false);

  tensor *output = tensor_matmul(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, output_shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Matmul failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, MATMUL);
  EXPECT_EQ(output->_backwards, _matmul_backwards);

  intarray_free(shape1);
  intarray_free(shape2);
  intarray_free(output_shape);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}

TEST(Tensor, Conv2d) {
  intarray *shape1 = intarray_build(4, 1, 2, 5, 5);
  float buffer1[] = {
      14.00,   16.0163, 37.04,   31.95,    26.4517,  -14.4,   -42.1,  -7.6143,
      20.696,  17.15,   15.8212, -46.89,   -22.6,    36.531,  -44.1,  10.89,
      5.6016,  3.22,    -17.64,  -18.9343, -42.07,   32.44,   49.348, 5.807,
      -27.1,   14.00,   16.0163, 37.04,    31.95,    26.4517, -14.4,  -42.1,
      -7.6143, 20.696,  17.15,   15.8212,  -46.89,   -22.6,   36.531, -44.1,
      10.89,   5.6016,  3.22,    -17.64,   -18.9343, -42.07,  32.44,  49.348,
      5.807,   -27.1};

  intarray *shape2 = intarray_build(4, 2, 2, 3, 3);
  float kernel_buffer[] = {
      14.00, 16.0163, 37.04, 17.15, 15.8212, -46.89, -17.64, -18.9343, -42.07,
      14.00, 16.0163, 37.04, 17.15, 15.8212, -46.89, -17.64, -18.9343, -42.07,
      14.00, 16.0163, 37.04, 17.15, 15.8212, -46.89, -17.64, -18.9343, -42.07,
      14.00, 16.0163, 37.04, 17.15, 15.8212, -46.89, -17.64, -18.9343, -42.07};

  intarray *output_shape = intarray_build(4, 1, 2, 3, 3);
  float correct_output[] = {
      5656.021484,  -187.643921,  5930.000977,  -2004.711426, -4474.199707,
      8384.197266,  -6380.788086, -884.161560,  -1082.318726, 5656.021484,
      -187.643921,  5930.000977,  -2004.711426, -4474.199707, 8384.197266,
      -6380.788086, -884.161560,  -1082.318726,
  };

  tensor *correct = tensor_from_buffer(output_shape, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);
  tensor *tensor2 = tensor_from_buffer(shape2, kernel_buffer, false);
  tensor *output = tensor_conv2d(tensor1, tensor2, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, output_shape));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Conv2d failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, CONV_2D);
  EXPECT_EQ(output->_backwards, _conv2d_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Square) {
  intarray *shape1 = intarray_build(2, 3, 3);
  float buffer1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  float correct_output[] = {0, 1, 4, 9, 16, 25, 36, 49, 64};

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_square(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Square failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SQUARE);
  EXPECT_EQ(output->_backwards, _square_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Sqrt) {
  intarray *shape1 = intarray_build(2, 3, 3);
  float buffer1[] = {0, 1, 4, 9, 16, 25, 36, 49, 64};

  float correct_output[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_sqrt(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Sqrt failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, SQRT);
  EXPECT_EQ(output->_backwards, _sqrt_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Exp) {
  intarray *shape1 = intarray_build(2, 3, 3);
  float buffer1[] = {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0};
  float correct_output[] = {
      1.000000,  1.648721,  2.718282,  4.481689,  7.389056,
      12.182494, 20.085537, 33.115452, 54.598148,
  };

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_exp(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Sqrt failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, EXP);
  EXPECT_EQ(output->_backwards, _exp_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Log) {
  intarray *shape1 = intarray_build(2, 3, 3);
  float buffer1[] = {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0};
  float correct_output[] = {
      -INFINITY, -0.693147, 0.000000, 0.405465, 0.693147,
      0.916291,  1.098612,  1.252763, 1.386294,
  };

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_log(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Log failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, LOG);
  EXPECT_EQ(output->_backwards, _log_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}

TEST(Tensor, Reciprocal) {
  intarray *shape1 = intarray_build(2, 3, 3);
  float buffer1[] = {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0};
  float correct_output[] = {
      INFINITY, 2.000000, 1.000000, 0.666667, 0.500000,
      0.400000, 0.333333, 0.285714, 0.250000,
  };

  tensor *correct = tensor_from_buffer(shape1, correct_output, false);
  tensor *tensor1 = tensor_from_buffer(shape1, buffer1, false);

  tensor *output = tensor_reciprocal(tensor1, false);

  ASSERT_TRUE(intarray_equal(output->vw->shape, shape1));

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Reciprocal failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  EXPECT_FALSE(output->grads);
  EXPECT_FALSE(output->requires_grad);
  EXPECT_FALSE(output->parents);
  EXPECT_EQ(output->op, RECIPROCAL);
  EXPECT_EQ(output->_backwards, _reciprocal_backwards);

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
}
