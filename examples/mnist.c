// model based off of
// https://github.com/pytorch/examples/blob/main/mnist/main.py
#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <stdint.h>
#include <time.h>

#define MNIST_IMAGE_SIZE 28
#define MNIST_TEST_SIZE 10000

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

typedef struct {
  tt *conv_layer_1;
  tt *conv_layer_2;
  tt *linear_layer_1;
  tt *linear_layer_2;
} mnist_cnn;

int main(void) {
  srand(time(NULL));
  int batch_size = envvar("BS", 1);

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
