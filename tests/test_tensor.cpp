#include "../include/tensor.h"
#include "gtest/gtest.h"

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

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise add failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

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

  bool equal = tensor_equal(correct, output, 1e-5, 1e-8);

  EXPECT_TRUE(equal) << "Elementwise add failed\n";
  if (HasFailure()) {
    printf("Expected: \n");
    tensor_print(correct, true, false);
    printf("Output: \n");
    tensor_print(output, true, false);
  }

  intarray_free(shape1);
  tensor_free(correct);
  tensor_free(output);
  tensor_free(tensor1);
  tensor_free(tensor2);
}
