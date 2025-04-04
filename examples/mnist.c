// model based off of
// htensorps://github.com/pytorch/examples/blob/main/mnist/main.py
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

void load_mnist_image(tensor *input_batch, tensor *output_batch,
                      const char *file_name, int file_length, int line) {
  assert(line < file_length);
  char *image = read_mnist_image(file_name, line);
  load_mnist_buffer(image, input_batch->data->buffer,
                    output_batch->data->buffer, 0, 0);
  free(image);
}

void load_mnist_batch(tensor *input_batch, tensor *output_batch,
                      const char *file_name, int file_length, int batch_size) {
  for (int i = 0; i < batch_size; i++) {
    int index = randi(0, file_length);
    char *image = read_mnist_image(file_name, index);
    load_mnist_buffer(
        image, input_batch->data->buffer, output_batch->data->buffer,
        i * intarray_prod(input_batch->vw->shape) / batch_size, i);
    free(image);
  }
}

void display_mnist_image(tensor *image, tensor *correct, tensor *guesses) {
  intarray *index = intarray_zeros(4);
  for (int b = 0; b < image->vw->shape->items[0]; b++) {
    index->items[0] = b;
    float answer = correct->data->buffer[b];
    if (image->vw->shape->items[0] > 1) {
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

    for (int h = 0; h < image->vw->shape->items[2]; h++) {
      index->items[2] = h;
      for (int w = 0; w < image->vw->shape->items[3]; w++) {
        index->items[3] = w;
        float value = tensor_getindex(image, index);
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
  intarray_free(index);
}

typedef struct {
  tensor *conv_layer_1;
  tensor *conv_layer_2;
  tensor *linear_layer_1;
  tensor *linear_layer_2;
} mnist_cnn;

void setup_mnist_model(mnist_cnn *model) {
  intarray *conv_layer_1_shape = intarray_build(4, 32, 1, 3, 3);
  model->conv_layer_1 = tensor_conv_init(conv_layer_1_shape, 1, 3, true);

  intarray *conv_layer_2_shape = intarray_build(4, 64, 32, 3, 3);
  model->conv_layer_2 = tensor_conv_init(conv_layer_2_shape, 32, 3, true);

  intarray *linear_layer_1_shape = intarray_build(2, 128, 9216);
  model->linear_layer_1 = tensor_linear_init(linear_layer_1_shape, 9216, true);

  intarray *linear_layer_2_shape = intarray_build(2, 10, 128);
  model->linear_layer_2 = tensor_linear_init(linear_layer_2_shape, 128, true);

  intarray_free(conv_layer_1_shape);
  intarray_free(conv_layer_2_shape);
  intarray_free(linear_layer_1_shape);
  intarray_free(linear_layer_2_shape);
}

void free_mnist_model(mnist_cnn *model) {
  tensor_free(model->conv_layer_1);
  tensor_free(model->conv_layer_2);
  tensor_free(model->linear_layer_1);
  tensor_free(model->linear_layer_2);
}

tensor *infer_mnist_model(mnist_cnn *model, tensor *input_batch) {
  int batch_size = input_batch->vw->shape->items[0];
  tensor *l1 = tensor_conv2d(input_batch, model->conv_layer_1, true);
  tensor *l1_activations = tensor_relu(l1, true);

  tensor *l2 = tensor_conv2d(l1_activations, model->conv_layer_2, true);
  tensor *l2_activations = tensor_relu(l2, true);

  tensor *maxpool = tensor_maxpool2d(l2_activations, 2, true);
  tensor *flatteneded_maxpool = flatten(maxpool, 1);

  const int convs_outputs_flattened =
      model->conv_layer_2->vw->shape->items[0] * 144;
  intarray *new_shape =
      intarray_build(3, batch_size, convs_outputs_flattened, 1);
  tensor *l3_input = tensor_reshape(flatteneded_maxpool, new_shape, true);
  intarray_free(new_shape);

  tensor *l3 = tensor_matmul(model->linear_layer_1, l3_input, true);
  tensor *l3_activations = tensor_relu(l3, true);
  tensor *l4 = tensor_matmul(model->linear_layer_2, l3_activations, true);
  tensor *logits = flatten(l4, 1);
  return logits;
}

float train_step_mnist_model(mnist_cnn *model, float learning_rate,
                             int batch_size, tensor *input_batch,
                             tensor *output_batch) {
  tensor *logits = infer_mnist_model(model, input_batch);
  tensor *log_probs = log_softmax(logits);
  tensor *loss = cross_entropy(log_probs, output_batch);
  float value = storage_getitem(loss->data, 0);

  graph *network = graph_build(loss);
  graph_zeroed(network);
  graph_backprop(network);
  sgd_step(network, learning_rate);
  graph_free(network);
  return value;
}

// TODO: after training check accuracy.
int main(void) {
  srand(time(NULL));
  const int batch_size = envvar("BS", 1);
  const float learning_rate = 1e-4f;
  const int steps = 100;


  intarray *input_batch_shape = intarray_build(4, batch_size, 1, 28, 28);
  intarray *output_batch_shape = intarray_build(1, batch_size);

  mnist_cnn model;
  setup_mnist_model(&model);

  // train
  tensor *input_batch = tensor_zeros(input_batch_shape, false);
  tensor *output_batch = tensor_zeros(output_batch_shape, false);
  for (int i = 0; i < steps; i++) {
    load_mnist_batch(input_batch, output_batch, "../data/mnist_test.csv", 10000,
                     batch_size);
    tensor *one_hot_encoded_output = one_hot_encode(output_batch, 10);
    float loss = train_step_mnist_model(&model, learning_rate, batch_size,
                                        input_batch, one_hot_encoded_output);
    printf("Loss: %f\n", loss);
  }

  tensor_free(input_batch);
  tensor_free(output_batch);
  intarray_free(input_batch_shape);
  intarray_free(output_batch_shape);
  free_mnist_model(&model);
}
