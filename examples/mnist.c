// model based off of
// https://github.com/pytorch/examples/blob/main/mnist/main.py
#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <math.h>
#include <stdint.h>
#include <time.h>

#define MNIST_IMAGE_SIZE 28
#define MNIST_TEST_SIZE 10000

int randi(int min, int max) {
  int value = roundf((float)rand() / (float)RAND_MAX * (max - min) + min);
  return value;
}

int envvar(const char *name, int default_value) {
  const char *var = getenv(name);
  if (var == NULL) {
    return default_value;
  }
  char *endptr;
  unsigned long result;
  result = strtoul(var, &endptr, 10);
  assert(*endptr == 0 && "Only accepts ints");
  return result;
}

char *read_mnist_image(const char *file_name, int line_number) {
  FILE *stream = fopen(file_name, "r");
  assert(stream != NULL && "Couldn't read file");
  char *line = NULL;
  size_t len = 0;
  ssize_t nread;

  int current_line = 0;

  while ((nread = getline(&line, &len, stream)) != -1) {
    if (current_line == line_number) {
      fclose(stream);
      return line;
    }
    current_line++;
  }
  fclose(stream);
  free(line);
  return NULL;
}

void load_mnist_buffer(const char *image, float *input, float *output,
                       int input_index, int output_index) {
  int len = strlen(image);
  float output_value = (float)(image[0] - '0');
  output[output_index] = output_value;

  int image_index = 0;
  int buffer_index = 0;
  const int offset = 2;

  while (image_index + offset < len) {
    float value = 0;
    char current = image[image_index + offset];
    while (current <= '9' && current >= '0') {
      value *= 10;
      value += (float)(current - '0');
      image_index++;
      current = image[image_index + offset];
    }
    input[input_index + buffer_index] = value;
    image_index++;
    buffer_index++;
  }
}

void load_mnist_image(tt *input_batch, tt *output_batch, const char *file_name,
                      int file_length, int line) {
  assert(line < file_length);
  char *image = read_mnist_image(file_name, line);
  load_mnist_buffer(image, input_batch->data->buffer,
                    output_batch->data->buffer, 0, 0);
  free(image);
}

void load_mnist_batch(tt *input_batch, tt *output_batch, const char *file_name,
                      int file_length, int batch_size) {
  for (int i = 0; i < batch_size; i++) {
    int index = randi(0, file_length);
    char *image = read_mnist_image(file_name, index);
    load_mnist_buffer(
        image, input_batch->data->buffer, output_batch->data->buffer,
        i * ttuple_prod(input_batch->view->shape) / batch_size, i);
    free(image);
  }
}

void display_mnist_image(tt *image, tt *correct, tt *guesses) {
  ttuple *index = ttuple_zeros(4);
  for (int b = 0; b < image->view->shape->items[0]; b++) {
    index->items[0] = b;
    float answer = correct->data->buffer[b];
    if (image->view->shape->items[0] > 1) {
      printf("Batch item #%d\n", b + 1);
    } else {
      printf("Single image\n");
    }
    printf("Correct Answer: %f", answer);
    if (guesses != NULL) {
      printf(", Guess: %f\n", guesses->data->buffer[b]);
    } else {
      printf("\n");
    }

    for (int h = 0; h < image->view->shape->items[2]; h++) {
      index->items[2] = h;
      for (int w = 0; w < image->view->shape->items[3]; w++) {
        index->items[3] = w;
        float value = tt_getindex(image, index);
        if (value > 150) {
          printf("MM");
        } else {
          printf("..");
        }
      }
      printf("\n");
    }
    printf("\n");
  }
  ttuple_free(index);
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

typedef struct {
  tt *conv_layer_1;
  tt *conv_layer_2;
  tt *linear_layer_1;
  tt *linear_layer_2;
} mnist_cnn;

int main(void) {
  srand(time(NULL));
  int batch_size = envvar("BS", 8);

  ttuple *input_batch_shape = ttuple_build(4, batch_size, 1, 28, 28);
  tt *input_batch = tt_zeros(input_batch_shape, false);

  ttuple *output_batch_shape = ttuple_build(2, batch_size, 1);
  tt *output_batch = tt_zeros(output_batch_shape, false);

  load_mnist_batch(input_batch, output_batch, "data/mnist_test.csv", 10000,
                   batch_size);
  // display_mnist_image(input_batch, output_batch, NULL);

  mnist_cnn model;

  ttuple *conv_layer_1_shape = ttuple_build(4, 32, 1, 3, 3);
  model.conv_layer_1 = tt_conv_init(conv_layer_1_shape, 1, 3, true);

  ttuple *conv_layer_2_shape = ttuple_build(4, 64, 32, 3, 3);
  model.conv_layer_2 = tt_conv_init(conv_layer_2_shape, 32, 3, true);

  ttuple *linear_layer_1_shape = ttuple_build(2, 128, 9216);
  model.linear_layer_1 = tt_linear_init(linear_layer_1_shape, 9216, true);

  ttuple *linear_layer_2_shape = ttuple_build(2, 10, 128);
  model.linear_layer_2 = tt_linear_init(linear_layer_2_shape, 128, true);

  // // Conv (Layer 1)
  tt *l1 = tt_conv2d(input_batch, model.conv_layer_1);
  tt *l1_activations = tt_relu(l1);
  // Conv (Layer 2)
  tt *l2 = tt_conv2d(l1_activations, model.conv_layer_2);
  tt *l2_activations = tt_relu(l2);
  // Maxpool + Flatten
  tt *maxpool = tt_maxpool2d(l2_activations, 2);
  tt *flattened_maxpool = flatten(maxpool, 1);
  ttuple *new_shape = ttuple_build(3, batch_size, 9216, 1);
  tt *l3_input = tt_reshape(flattened_maxpool, new_shape);
  // Linear (Layer 3)
  tt *l3 = tt_matmul(model.linear_layer_1, l3_input);
  tt *l3_activations = tt_relu(l3);
  // Linear (Layer 3)
  tt *l4 = tt_matmul(model.linear_layer_2, l3_activations);
  tt *logits = flatten(l4, 1);
  tt_print(logits, true, false);
  // // log softmax
  // tt *log_probs = log_softmax(logits);
  // tt_print(log_probs, true, false);
}
