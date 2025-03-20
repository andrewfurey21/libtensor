#include "gtest/gtest.h"
#include "../include/tensor.h"

TEST(Tensor, Add) {
    intarray* shape1 = intarray_build(4, 1, 2, 3, 4);
    float* buffer = (float*)malloc(sizeof(float) * 2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; i++) {
        buffer[i] = (float)i;
    }
    tensor* tensor1 = tensor_from_buffer(shape1, buffer, false);
    tensor* tensor2 = tensor_from_buffer(shape1, buffer, false);
}
