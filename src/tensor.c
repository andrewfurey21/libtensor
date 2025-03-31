#include "assert.h"
#include "malloc.h"
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include <math.h>
#include <stdint.h>

#include "../include/tensor.h"

#define max(a, b) ((a) > (b) ? a : b)
#define min(a, b) ((a) < (b) ? a : b)

tensor *setup_tensor(view *vw, storage *data, bool requires_grad,
                     tensor **parents, tensor_op op,
                     void (*pfn_backprop)(tensor *self)) {

  uint64_t size = intarray_prod(vw->shape);
  assert(size >= 0);

  tensor *t = (tensor *)malloc(sizeof(tensor));
  tensor *grads = NULL;
  t->vw = vw;

  if (requires_grad) {
    grads = tensor_zeros(vw->shape, false);
  }

  t->data = data;
  t->requires_grad = requires_grad;
  t->parents = parents;
  t->op = op;
  t->grads = grads;
  t->_backwards = pfn_backprop;
  return t;
}

tensor *tensor_zeros(intarray *shape, bool requires_grad) {
  uint64_t size = intarray_prod(shape);
  storage *data = storage_zeros(size);
  view *vw = view_new(shape);
  return setup_tensor(vw, data, requires_grad, NULL, NOOP, NULL);
}

tensor *tensor_ones(intarray *shape, bool requires_grad) {
  tensor *ones = tensor_zeros(shape, requires_grad);
  for (size_t i = 0; i < ones->data->size; i++) {
    storage_setitem(ones->data, i, 1.0f);
  }
  return ones;
}

tensor *tensor_from_buffer(intarray *shape, float *buffer, bool requires_grad) {
  uint64_t size = intarray_prod(shape);
  storage *data = storage_from_buffer(size, buffer);
  view *vw = view_new(shape);
  return setup_tensor(vw, data, requires_grad, NULL, NOOP, NULL);
}

float tensor_getindex(tensor *input, intarray *index) {
  uint64_t physical_index = view_index(input->vw, index);
  return storage_getitem(input->data, physical_index);
}
void tensor_setindex(tensor *input, intarray *index, float num) {
  uint64_t physical_index = view_index(input->vw, index);
  storage_setitem(input->data, physical_index, num);
}

tensor *tensor_fill(intarray *shape, float fill_value, bool requires_grad) {
  tensor *t = tensor_zeros(shape, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    storage_setitem(t->data, i, fill_value);
  }
  return t;
}

tensor *tensor_linspace(intarray *shape, float min, float max,
                        bool requires_grad) {
  int steps = intarray_prod(shape);
  tensor *t = tensor_zeros(shape, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = min + i * ((max - min) / (steps - 1));
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_uniform(intarray *shape, float min, float max,
                       bool requires_grad) {
  tensor *t = tensor_zeros(shape, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (float)rand() / (float)RAND_MAX * (max - min) + min;
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_uniformint(intarray *shape, float min, float max,
                          bool requires_grad) {
  tensor *t = tensor_uniform(shape, min, max, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = round(storage_getitem(t->data, i));
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_copy(tensor *original, bool requires_grad) {
  storage *data = storage_copy(original->data);
  view *vw = view_copy(original->vw);
  return setup_tensor(vw, data, requires_grad, NULL, NOOP, NULL);
}

void tensor_to_zeros(tensor *input) { storage_to_zeros(input->data); }

void tensor_to_n(struct tensor *input, float n) {
  for (int i = 0; i < input->data->size; i++) {
    storage_setitem(input->data, i, n);
  }
}

void tensor_print(tensor *input, bool show_buffer, bool show_grads) {
  if (!input) {
    printf("values: (null)\n");
    return;
  }
  intarray_print(input->vw->shape);
  if (input->requires_grad) {
    print_tensor_op(input->op);
  }
  if (show_buffer) {
    printf("values: [ ");
    for (int i = 0; i < input->data->size; i++) {
      printf("%f, ", storage_getitem(input->data, i));
    }
    printf("]\n");
  }
  if (input->requires_grad && show_grads) {
    printf("gradient shape: ");
    intarray_print(input->grads->vw->shape);
    printf("gradient values: [ ");
    for (int i = 0; i < input->grads->data->size; i++) {
      printf("%f, ", storage_getitem(input->grads->data, i));
    }
    printf("]\n");
  }
}

void tensor_free(tensor *input) {
  storage_free(input->data);
  view_free(input->vw);
  free(input->parents);
  if (input->requires_grad) {
    tensor_free(input->grads); // make sure grads cant have grads
  }
  free(input);
}

bool tensor_equal(tensor *a, tensor *b, float rtol, float atol) {
  assert(view_equal(a->vw, b->vw));
  for (int i = 0; i < intarray_prod(a->vw->shape); i++) {
    float b_val = storage_getitem(b->data, i);
    float a_val = storage_getitem(a->data, i);
    if (fabs(a_val - b_val) > atol + rtol * fabs(b_val)) {
      return false;
    }
  }
  return true;
}

tensor *tensor_linear_init(intarray *shape, int in_features,
                           bool requires_grad) {
  float bound = 1.0f / sqrtf((float)in_features);
  return tensor_uniform(shape, -bound, bound, requires_grad);
}

tensor *tensor_conv_init(intarray *shape, int in_channels, int kernel_size,
                         bool requires_grad) {
  float scale = 1.0f / sqrtf((float)(in_channels * kernel_size * kernel_size));
  return tensor_uniform(shape, -scale, scale, requires_grad);
}

void _add_backwards(tensor *self) {
  if (self->parents[0]->requires_grad) {
    tensor *grads_0 = tensor_add(self->grads, self->parents[0]->grads, false);
    tensor_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tensor *grads_1 = tensor_add(self->grads, self->parents[1]->grads, false);
    tensor_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}

tensor *tensor_add(tensor *a, tensor *b, bool track_grads) {
  assert(view_equal(a->vw, b->vw) &&
         "Tensors are not the same shape.");
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(ADD) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }
  storage *data = storage_zeros(a->data->size);
  view *vw = view_copy(a->vw);
  for (uint64_t i = 0; i < a->data->size; i++) {
    float value = storage_getitem(a->data, i) + storage_getitem(b->data, i);
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, ADD, _add_backwards);
}

void _sub_backwards(tensor *self) {
  if (self->parents[0]->requires_grad) {
    tensor *grads_0 = tensor_add(self->grads, self->parents[0]->grads, false);
    tensor_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tensor *grads_1 = tensor_sub(self->parents[1]->grads, self->grads, false);
    tensor_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}

tensor *tensor_sub(tensor *a, tensor *b, bool track_grads) {
  assert(view_equal(a->vw, b->vw) &&
         "Tensors are not the same shape.");
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(SUB) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }

  storage *data = storage_zeros(a->data->size);
  view *vw = view_copy(a->vw);
  for (uint64_t i = 0; i < a->data->size; i++) {
    float value = storage_getitem(a->data, i) - storage_getitem(b->data, i);
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, SUB, _sub_backwards);
}

void _mul_backwards(tensor *self) {
  if (self->parents[0]->requires_grad) {
    tensor *grads_0 = tensor_mul(self->grads, self->parents[1], false);
    tensor *acc_grads_0 = tensor_add(grads_0, self->parents[0]->grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(grads_0);
    self->parents[0]->grads = acc_grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tensor *grads_1 = tensor_mul(self->grads, self->parents[0], false);
    tensor *acc_grads_1 = tensor_add(grads_1, self->parents[1]->grads, false);
    tensor_free(self->parents[1]->grads);
    tensor_free(grads_1);
    self->parents[1]->grads = acc_grads_1;
  }
}

tensor *tensor_mul(tensor *a, tensor *b, bool track_grads) {
  assert(view_equal(a->vw, b->vw) &&
         "Tensors are not the same shape.");
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(MUL) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }
  storage *data = storage_zeros(a->data->size);
  view *vw = view_copy(a->vw);
  for (uint64_t i = 0; i < a->data->size; i++) {
    float value = storage_getitem(a->data, i) * storage_getitem(b->data, i);
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, MUL, _mul_backwards);
}

void _sum_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  intarray *unit_shape = intarray_build(1, 1);

  intarray *self_shape = self->vw->shape;
  intarray *parent_shape = self->parents[0]->vw->shape;
  if (intarray_equal(unit_shape, self_shape)) {
    tensor *expanded_grads =
        tensor_fill(parent_shape, storage_getitem(self->grads->data, 0), false);
    tensor *acc_grads =
        tensor_add(self->parents[0]->grads, expanded_grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  } else {
    int expand_axis = 0;
    assert(self_shape->size == parent_shape->size);

    // NOTE: this might not work
    for (int i = 0; i < self_shape->size; i++) {
      if (self_shape->items[i] == 1 && parent_shape->items[i] != 1) {
        expand_axis = i;
        break;
      }
    }

    tensor *expanded_grads = tensor_zeros(parent_shape, false);

    intarray *current = intarray_zeros(parent_shape->size);
    uint64_t along_axis = parent_shape->items[expand_axis];
    for (uint64_t i = 0; i < self->grads->data->size; i++) {
      // expanding
      for (uint64_t j = 0; j < along_axis; j++) {
        intarray *current_grads = intarray_copy(current);
        current_grads->items[expand_axis] = 0;
        float num = tensor_getindex(self->grads, current_grads);
        tensor_setindex(expanded_grads, current, num);
        current->items[expand_axis]++;
        intarray_free(current_grads);
      }

      current->items[expand_axis] = 0;
      // updating current (with expanded axis set to 0)
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == expand_axis) {
          continue;
        }
        current->items[k]++;
        if (current->items[k] >= parent_shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }

    tensor *acc_grads =
        tensor_add(self->parents[0]->grads, expanded_grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  }
  intarray_free(unit_shape);
}

// axis=-1 => sum up all elements
// currently always keepdims, except for axis=-1
// could seriously use some tests here
tensor *tensor_sum(tensor *input, int axis, bool track_grads) {
  assert(axis >= -1 && axis < (int)input->vw->shape->size);
  intarray *new_shape;
  if (axis == -1) {
    new_shape = intarray_build(1, 1);
  } else {
    new_shape = intarray_copy(input->vw->shape);
    new_shape->items[axis] = 1;
  }

  bool requires_grad = input->requires_grad && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents =
        (tensor **)malloc(tensor_op_operands(SUM_REDUCE) * sizeof(tensor *));
    parents[0] = input;
  }

  storage *data = storage_zeros(intarray_prod(new_shape));
  view *vw = view_new(new_shape);

  if (axis == -1) {
    double sum = 0.0f;
    for (uint64_t i = 0; i < input->data->size; i++) {
      sum += storage_getitem(input->data, i);
    }
    storage_setitem(data, 0, sum);
  } else {
    uint64_t along_axis = input->vw->shape->items[axis];
    uint64_t num_accumulate = intarray_prod(input->vw->shape) / along_axis;
    intarray *current = intarray_zeros(input->vw->shape->size);
    for (uint64_t i = 0; i < num_accumulate; i++) {
      float sum = 0.0f;
      for (uint64_t j = 0; j < along_axis; j++) {
        sum += storage_getindex(input->data, input->vw, current);
        current->items[axis]++;
      }
      current->items[axis] = 0;
      storage_setindex(data, vw, current, sum);
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == axis)
          continue;
        current->items[k]++;
        if (current->items[k] >= input->vw->shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }
  }

  intarray_free(new_shape);
  return setup_tensor(vw, data, requires_grad, parents, SUM_REDUCE,
                      _sum_backwards);
}

void _relu_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }

  tensor *grads = tensor_zeros(self->vw->shape, false);
  for (size_t i = 0; i < self->parents[0]->data->size; i++) {
    if (storage_getitem(self->parents[0]->data, i) > 0) {
      storage_setitem(grads->data, i, 1);
    }
  }
  tensor *mul_grads = tensor_mul(self->grads, grads, false);
  tensor *acc_grads = tensor_add(self->parents[0]->grads, mul_grads, false);
  tensor_free(grads);
  tensor_free(self->parents[0]->grads);
  tensor_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_relu(tensor *input, bool track_grads) {
  tensor **parents = NULL;
  bool requires_grad = input->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(RELU) * sizeof(tensor *));
    parents[0] = input;
  }

  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = storage_getitem(input->data, i);
    storage_setitem(data, i, value * (value > 0));
  }

  return setup_tensor(vw, data, requires_grad, parents, RELU, _relu_backwards);
}

// Reshape
void _reshape_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad)
    return;
  tensor *grads =
      tensor_reshape(self->grads, self->parents[0]->vw->shape, false);
  tensor *acc_grads = tensor_add(grads, self->parents[0]->grads, false);
  tensor_free(grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_reshape(tensor *a, intarray *new_shape, bool track_grads) {
  assert(intarray_prod(new_shape) == intarray_prod(a->vw->shape));
  tensor **parents = NULL;
  bool requires_grad = a->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(RESHAPE) * sizeof(tensor *));
    parents[0] = a;
  }
  storage *data = storage_copy(a->data);
  view *vw = view_new(new_shape);
  return setup_tensor(vw, data, requires_grad, parents, RESHAPE,
                      _reshape_backwards);
}

// Expand
// basically forwards sum (could totally do a refactor)
void _expand_backwards(tensor *self) {
  // sum
  // shape should be same for tensor and their gradients
  intarray *div = intarray_div(self->vw->shape, self->parents[0]->vw->shape);
  int expanded_axis = -1;
  for (int i = 0; i < div->size; i++) {
    if (div->items[i] != 1) {
      expanded_axis = i;
      break;
    }
  }
  assert(expanded_axis != -1 &&
         "Did not find an expanded axis from self->vw->shape");

  // sum self->parents[0]->grads along expanded_axis

  uint64_t along_axis = self->vw->shape->items[expanded_axis];
  uint64_t num_accumulate = intarray_prod(self->vw->shape) / along_axis;
  intarray *current = intarray_zeros(self->vw->shape->size);

  tensor *self_grad = self->grads;
  tensor *parent_grad = self->parents[0]->grads;
  for (uint64_t i = 0; i < num_accumulate; i++) {
    float sum = 0.0f;
    for (uint64_t j = 0; j < along_axis; j++) {
      sum += tensor_getindex(self_grad, current);
      current->items[expanded_axis]++;
    }
    current->items[expanded_axis] = 0;
    tensor_setindex(parent_grad, current, sum);
    for (int k = current->size - 1; k >= 0; k--) {
      if (k == expanded_axis)
        continue;
      current->items[k]++;
      if (current->items[k] >= self->vw->shape->items[k]) {
        current->items[k] = 0;
        continue;
      }
      break;
    }
  }
}

// can expand axis where dim>=1
// basically backwards sum
// follows broadcasting rules, cannot expand dim that isn't 1
tensor *tensor_expand(tensor *input, uint64_t axis, uint64_t factor,
                      bool track_grads) {
  intarray *new_shape = intarray_copy(input->vw->shape);
  assert(axis >= 0 && axis < new_shape->size &&
         "Axis to expand is out of range.");
  assert(factor > 0 && "Expanding factor must be greater than 0");
  assert(new_shape->items[axis] == 1 && "Cannot expand [axis]!=1");

  // calculate new shape here
  new_shape->items[axis] *= factor;

  bool requires_grad = track_grads && input->requires_grad;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(EXPAND) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(intarray_prod(new_shape));
  view *vw = view_new(new_shape);

  intarray *expanded_index = intarray_zeros(vw->shape->size);
  uint64_t along_axis = new_shape->items[axis];
  for (uint64_t i = 0; i < data->size; i++) {
    for (uint64_t j = 0; j < along_axis; j++) {
      intarray *original_index = intarray_copy(expanded_index);
      original_index->items[axis] = 0;
      float num = tensor_getindex(input, original_index);
      storage_setindex(data, vw, expanded_index, num);
      expanded_index->items[axis]++;
      intarray_free(original_index);
    }
    expanded_index->items[axis] = 0;
    for (int k = expanded_index->size - 1; k >= 0; k--) {
      if (k == axis) {
        continue;
      }
      expanded_index->items[k]++;
      if (expanded_index->items[k] >= input->vw->shape->items[k]) {
        expanded_index->items[k] = 0;
        continue;
      }
      break;
    }
  }

  intarray_free(new_shape);
  return setup_tensor(vw, data, requires_grad, parents, EXPAND,
                      _expand_backwards);
}

void _neg_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tensor *grads = tensor_fill(self->vw->shape, -1.0f, false);
  tensor *mul_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads = tensor_add(mul_grads, self->parents[0]->grads, false);
  tensor_free(self->parents[0]->grads);
  tensor_free(grads);
  tensor_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_neg(tensor *input, bool track_grads) {
  bool requires_grad = track_grads && input->requires_grad;
  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  tensor **parents = NULL;
  if (track_grads) {
    parents = (tensor **)malloc(tensor_op_operands(NEG) * sizeof(tensor *));
    parents[0] = input;
  }
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = storage_getitem(input->data, i);
    storage_setitem(data, i, -value);
  }
  return setup_tensor(vw, data, requires_grad, parents, NEG, _neg_backwards);
}

// still assuming square kernel
void _maxpool2d_backwards(tensor *self) {
  // parent grads will be tensor, only with 1s or 0s
  // so basically max pool, except keep track of max index.
  // set max index to 1, others to 0
  // mul expanded by sparse grads
  // then acc to parents->grads.

  intarray *self_shape = self->vw->shape;
  intarray *parent_shape = self->parents[0]->vw->shape;
  int x_index = self_shape->size - 1;

  int pooled_width = self_shape->items[x_index];
  int original_width = parent_shape->items[x_index];
  int pooled_height = self_shape->items[x_index - 1];
  int original_height = parent_shape->items[x_index - 1];
  assert(original_width / pooled_width == original_height / pooled_height &&
         "not a square kernel buddy.");

  int kernel_size = original_width / pooled_width;

  int dims = parent_shape->size;
  int channels = parent_shape->size > 2 ? parent_shape->items[dims - 3] : 0;
  int batches = parent_shape->size > 3 ? parent_shape->items[dims - 4] : 0;

  // expanding self->grads
  tensor *expanded_self_grad = tensor_zeros(parent_shape, false);
  intarray *index = intarray_zeros(parent_shape->size);
  // i dont like 5 nested for loops :(
  for (int b = 0; b < fmax(batches, 1); b++) {
    if (batches)
      index->items[x_index - 3] = b;
    for (int c = 0; c < fmax(channels, 1); c++) {
      if (channels)
        index->items[x_index - 2] = c;
      for (int oh = 0; oh < original_height; oh++) {
        for (int ow = 0; ow < original_width; ow++) {
          index->items[x_index - 1] = oh / kernel_size;
          index->items[x_index] = ow / kernel_size;
          float value = tensor_getindex(self->grads, index);

          index->items[x_index] = ow;
          index->items[x_index - 1] = oh;
          tensor_setindex(expanded_self_grad, index, value);
        }
      }
    }
  }
  intarray_free(index);

  index = intarray_zeros(parent_shape->size);
  tensor *pooled_grads = tensor_ones(self->parents[0]->vw->shape, false);
  for (int b = 0; b < fmax(batches, 1); b++) {
    if (batches)
      index->items[x_index - 3] = b;
    for (int c = 0; c < fmax(channels, 1); c++) {
      if (channels)
        index->items[x_index - 2] = c;
      for (int oh = 0; oh < original_height; oh += kernel_size) {
        for (int ow = 0; ow < original_width; ow += kernel_size) {
          float max = -INFINITY;
          intarray *max_index = intarray_copy(index);
          for (int k = 0; k < kernel_size * kernel_size; k++) {
            int x = ow + (k % kernel_size);
            int y = oh + (k / kernel_size);
            index->items[x_index - 1] = y;
            index->items[x_index] = x;
            float value = tensor_getindex(
                self->parents[0],
                index);        // backprop is dependent on tensors staying
            if (value > max) { // if equal, its the first one.
              if (max != -INFINITY)
                tensor_setindex(pooled_grads, max_index, 0);
              max = value;
              intarray_free(max_index);
              max_index = intarray_copy(index);
            } else {
              tensor_setindex(pooled_grads, index, 0);
            }
          }
          intarray_free(max_index);
        }
      }
    }
  }
  intarray_free(index);

  tensor *expanded_by_pooled_grads =
      tensor_mul(expanded_self_grad, pooled_grads, false);
  tensor *accumulated_grads =
      tensor_add(self->parents[0]->grads, expanded_by_pooled_grads, false);
  tensor_free(expanded_self_grad);
  tensor_free(pooled_grads);
  self->parents[0]->grads = accumulated_grads;
}

// NOTE:
// assuming input is divisible by kernel size
// stride is kernel size
// no dilation, padding. ceilmode=False.
// 4d, 3d, 2d only.
tensor *tensor_maxpool2d(tensor *input, int kernel_size, bool track_grads) {
  intarray *input_shape = input->vw->shape;
  int x_index = input_shape->size - 1;

  assert(input_shape->size >= 2);
  assert(kernel_size > 1 && "Kernel size must be greater than 1");
  assert(input_shape->items[x_index] % kernel_size == 0 &&
         "Width not divisble by kernel size");
  assert(input_shape->items[x_index - 1] % kernel_size == 0 &&
         "Height not divisble by kernel size");

  int end_index = input_shape->size - 1;
  int original_width = input_shape->items[end_index];
  int original_height = input_shape->items[end_index - 1];

  int new_width = original_width / kernel_size;
  int new_height = original_height / kernel_size;

  intarray *new_shape = intarray_copy(input_shape);
  new_shape->items[end_index] = new_width;
  new_shape->items[end_index - 1] = new_height;

  tensor **parents = NULL;
  bool requires_grad = input->requires_grad && track_grads;
  if (requires_grad) {
    parents =
        (tensor **)malloc(tensor_op_operands(MAX_POOL_2D) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(intarray_prod(new_shape));
  view *vw = view_new(new_shape);

  int dims = input_shape->size;
  int channels = input_shape->size > 2 ? input_shape->items[dims - 3] : 0;
  int batches = input_shape->size > 3 ? input_shape->items[dims - 4] : 0;

  intarray *index = intarray_copy(input_shape);
  for (int b = 0; b < fmax(batches, 1); b++) {
    if (batches)
      index->items[x_index - 3] = b;
    for (int c = 0; c < fmax(channels, 1); c++) {
      if (channels)
        index->items[x_index - 2] = c;
      for (int oh = 0; oh < original_height; oh += kernel_size) {
        for (int ow = 0; ow < original_width; ow += kernel_size) {
          float max = -INFINITY;
          for (int k = 0; k < kernel_size * kernel_size; k++) {
            int x = ow + (k % kernel_size);
            int y = oh + (k / kernel_size);
            index->items[x_index - 1] = y;
            index->items[x_index] = x;
            float value = tensor_getindex(input, index);
            if (value > max)
              max = value;
          }
          int x = ow / kernel_size;
          int y = oh / kernel_size;
          index->items[x_index - 1] = y;
          index->items[x_index] = x;
          storage_setindex(data, vw, index, max);
        }
      }
    }
  }
  intarray_free(index);
  intarray_free(new_shape);
  return setup_tensor(vw, data, requires_grad, parents, MAX_POOL_2D,
                      _maxpool2d_backwards);
}

// really need broadcasting
// really hacky impl
// swaps last two dims
tensor *_transpose(tensor *original) {
  tensor *new = tensor_zeros(original->vw->shape, false);
  intarray *shape = new->vw->shape;

  int bs = shape->size >= 4 ? shape->items[shape->size - 4] : 1;
  int cs = shape->size >= 3 ? shape->items[shape->size - 3] : 1;
  int hs = shape->items[shape->size - 2];
  int ws = shape->items[shape->size - 1];

  assert(shape->size >= 2 && "can't transpose");
  int oldw = shape->items[shape->size - 1];
  shape->items[shape->size - 1] = shape->items[shape->size - 2];
  shape->items[shape->size - 2] = oldw;

  intarray *oi = intarray_zeros(MAX_ITEMS);
  intarray *ni = intarray_zeros(MAX_ITEMS);

  for (int b = 0; b < bs; b++) {
    oi->items[0] = b;
    ni->items[0] = b;

    oi->items[1] = 0;
    ni->items[1] = 0;
    for (int c = 0; c < cs; c++) {
      oi->items[1] = c;
      ni->items[1] = c;

      oi->items[2] = 0;
      ni->items[3] = 0;
      for (int h = 0; h < hs; h++) {
        oi->items[2] = h;
        ni->items[3] = h;

        oi->items[3] = 0;
        ni->items[2] = 0;
        for (int w = 0; w < ws; w++) {
          oi->items[3] = w;
          ni->items[2] = w;
          float value = tensor_getindex(original, oi);
          tensor_setindex(new, ni, value);
        }
      }
    }
  }
  intarray_free(oi);
  intarray_free(ni);
  return new;
}

// NOTE: optimization: pad once, then unpad.
// so shouldn't be doing pad here.
void _matmul_backwards(tensor *self) {
  intarray *wshape = self->parents[0]->vw->shape;
  intarray *ishape = self->parents[1]->vw->shape;

  int ww = self->parents[0]->vw->shape->items[wshape->size - 1];
  int wh = self->parents[0]->vw->shape->items[wshape->size - 2];

  int bw = self->parents[1]->vw->shape->items[ishape->size - 1];
  int bh = self->parents[1]->vw->shape->items[ishape->size - 2];
  int bs = ishape->size == 3 ? self->parents[1]->vw->shape->items[0] : 1;

  assert(ww == bh && "matmul can't do backprop on this shape");
  // weights: (inputs * transpose self.grad) sum, add
  if (self->parents[0]->requires_grad) {
    tensor *i = self->parents[1];
    tensor *grads_t = _transpose(self->grads);
    tensor *mm = tensor_matmul(i, grads_t, false);
    tensor *sum_grads = tensor_sum(mm, 0, false);
    tensor *grads = _transpose(sum_grads);

    intarray *squeezed_shape = intarray_squeeze(grads->vw->shape);
    intarray_free(grads->vw->shape);
    grads->vw->shape = squeezed_shape;

    tensor *acc_grads = tensor_add(grads, self->parents[0]->grads, false);
    tensor_free(grads);
    tensor_free(mm);
    tensor_free(grads_t);
    tensor_free(sum_grads);
    tensor_free(self->parents[0]->grads);
    self->parents[0]->grads = acc_grads;
  }

  // inputs: (transpose weights * self.grad) expand, add
  if (self->parents[1]->requires_grad) {
    tensor *w_t = _transpose(self->parents[0]);
    tensor *grads = tensor_matmul(w_t, self->grads, false);
    tensor *acc_grads = tensor_add(grads, self->parents[1]->grads, false);
    tensor_free(grads);
    tensor_free(self->parents[1]->grads);
    self->parents[1]->grads = acc_grads;
  }
}

// for now, inputs will be 2d or 3d
// shouldn't be that hard to generalize
tensor *tensor_matmul(tensor *a, tensor *b, bool track_grads) {
  int asize = a->vw->shape->size;
  int bsize = b->vw->shape->size;
  assert(asize == 2 || asize == 3);
  assert(bsize == 2 || bsize == 3);

  int aw = a->vw->shape->items[asize - 1];
  int ah = a->vw->shape->items[asize - 2];

  int bw = b->vw->shape->items[bsize - 1];
  int bh = b->vw->shape->items[bsize - 2];

  int as = asize == 3 ? a->vw->shape->items[0] : 1;
  int bs = bsize == 3 ? b->vw->shape->items[0] : 1;

  assert(aw == bh && "Tensors are not the correct shape");
  assert(as == bs || as == 1 ||
         bs == 1 && "Tensors don't have correct batch size");

  intarray *new_shape = intarray_build(3, max(as, bs), ah, bw);

  tensor **parents = NULL;
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(MATMUL) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }
  storage *data = storage_zeros(intarray_prod(new_shape));
  view *vw = view_new(new_shape);

  intarray *ai = intarray_zeros(3);
  intarray *bi = intarray_zeros(3);

  intarray *oi = intarray_zeros(3);

  for (int batch = 0; batch < bs; batch++) {
    oi->items[0] = batch;

    ai->items[0] = min(batch, as - 1);
    bi->items[0] = min(batch, bs - 1);

    for (int k = 0; k < ah; k++) {
      oi->items[1] = k;
      ai->items[1] = k;
      for (int j = 0; j < bw; j++) {
        oi->items[2] = j;
        bi->items[2] = j;
        float sum = 0;
        for (int i = 0; i < aw; i++) {
          bi->items[1] = i;
          ai->items[2] = i;
          float av = tensor_getindex(a, ai);
          float bv = tensor_getindex(b, bi);
          sum += av * bv;
        }
        storage_setindex(data, vw, oi, sum);
      }
    }
  }
  intarray_free(ai);
  intarray_free(bi);
  intarray_free(oi);
  intarray_free(new_shape);
  return setup_tensor(vw, data, requires_grad, parents, MATMUL,
                      _matmul_backwards);
}

void _conv2d_backwards(tensor *self) {
  int batch_size = self->vw->shape->items[0];
  int cout = self->vw->shape->items[1];
  int cin = self->parents[0]->vw->shape->items[1];
  int win = self->parents[0]->vw->shape->items[3];
  int hin = self->parents[0]->vw->shape->items[2];
  int kernel_size = self->parents[1]->vw->shape->items[3];

  // input gradients
  if (self->parents[0]->requires_grad) {
    tensor *grads = tensor_zeros(self->parents[0]->vw->shape, false);
    intarray *input_grad_index = intarray_zeros(4);
    intarray *kernel_index = intarray_zeros(4);
    intarray *output_grad_index = intarray_zeros(4);
    for (int b = 0; b < batch_size; b++) {
      input_grad_index->items[0] = b;
      output_grad_index->items[0] = b;
      for (int co = 0; co < cout; co++) {
        output_grad_index->items[1] = co;
        kernel_index->items[0] = co;
        for (int h = 0; h < hin - kernel_size + 1; h++) {
          output_grad_index->items[2] = h;
          for (int w = 0; w < win - kernel_size + 1; w++) {
            output_grad_index->items[3] = w;
            for (int ci = 0; ci < cin; ci++) {
              input_grad_index->items[1] = ci;
              kernel_index->items[1] = ci;
              for (int k = 0; k < kernel_size * kernel_size; k++) {
                int kh = k / kernel_size;
                int kw = k % kernel_size;

                input_grad_index->items[2] = h + kh;
                input_grad_index->items[3] = w + kw;

                kernel_index->items[2] = kh;
                kernel_index->items[3] = kw;

                float current_value = tensor_getindex(grads, input_grad_index);
                float kernel_value =
                    tensor_getindex(self->parents[1], kernel_index);
                float output_grad_value =
                    tensor_getindex(self->grads, output_grad_index);
                float new_value =
                    kernel_value * output_grad_value + current_value;
                tensor_setindex(grads, input_grad_index, new_value);
              }
            }
          }
        }
      }
    }
    tensor *acc_grads = tensor_add(grads, self->parents[0]->grads, false);

    tensor_free(grads);
    tensor_free(self->parents[0]->grads);

    intarray_free(kernel_index);
    intarray_free(input_grad_index);
    intarray_free(output_grad_index);

    self->parents[0]->grads = acc_grads;
  }

  // kernel gradients
  if (self->parents[1]->requires_grad) {
    tensor *grads = tensor_zeros(self->parents[1]->vw->shape, false);
    intarray *input_index = intarray_zeros(4);
    intarray *kernel_grad_index = intarray_zeros(4);
    intarray *output_grad_index = intarray_zeros(4);
    for (int b = 0; b < batch_size; b++) {
      input_index->items[0] = b;
      output_grad_index->items[0] = b;
      for (int co = 0; co < cout; co++) {
        output_grad_index->items[1] = co;
        kernel_grad_index->items[0] = co;
        for (int h = 0; h < hin - kernel_size + 1; h++) {
          output_grad_index->items[2] = h;
          for (int w = 0; w < win - kernel_size + 1; w++) {
            output_grad_index->items[3] = w;
            for (int ci = 0; ci < cin; ci++) {
              kernel_grad_index->items[1] = ci;
              input_index->items[1] = ci;
              for (int k = 0; k < kernel_size * kernel_size; k++) {
                int kh = k / kernel_size;
                int kw = k % kernel_size;

                input_index->items[2] = h + kh;
                input_index->items[3] = w + kw;

                kernel_grad_index->items[2] = kh;
                kernel_grad_index->items[3] = kw;

                float input_value =
                    tensor_getindex(self->parents[0], input_index);
                float current_value = tensor_getindex(grads, kernel_grad_index);
                float output_grad_value =
                    tensor_getindex(self->grads, output_grad_index);
                float new_value =
                    input_value * output_grad_value + current_value;
                tensor_setindex(grads, kernel_grad_index, new_value);
              }
            }
          }
        }
      }
    }
    tensor *acc_grads = tensor_add(grads, self->parents[1]->grads, false);

    tensor_free(grads);
    tensor_free(self->parents[1]->grads);

    intarray_free(kernel_grad_index);
    intarray_free(input_index);
    intarray_free(output_grad_index);

    self->parents[1]->grads = acc_grads;
  }
}

// dilation = 1, stride = 1, padding = 0,
// bias = false (just use add if you want a bias
// works only for input=4d, kernels=4d
// kernel shape: (cout, cin, kernelsize, kernelsize)
// input shape: (batch size, cin, hin, win)
// output shape: (batch size, cout, hout, wout)
// should add groups
tensor *tensor_conv2d(tensor *input, tensor *kernels, bool track_grads) {
  intarray *input_shape = input->vw->shape;
  assert(input_shape->size == 4);

  intarray *kernels_shape = kernels->vw->shape;
  assert(kernels_shape->size == 4);

  assert(kernels_shape->items[1] == input_shape->items[1]);
  // must be square
  assert(kernels_shape->items[2] == kernels_shape->items[3]);

  int batch_size = input_shape->items[0];
  int cout = kernels_shape->items[0];
  int cin = input_shape->items[1];
  int kernel_size = kernels_shape->items[3];

  int hin = input_shape->items[2];
  int win = input_shape->items[3];

  int wout = win - kernel_size + 1;
  int hout = hin - kernel_size + 1;

  intarray *out_shape = intarray_build(4, batch_size, cout, hout, wout);

  tensor **parents = NULL;
  bool requires_grad =
      (input->requires_grad || kernels->requires_grad) && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(CONV_2D) * sizeof(tensor *));
    parents[0] = input;
    parents[1] = kernels;
  }

  storage *data = storage_zeros(intarray_prod(out_shape));
  view *vw = view_new(out_shape);

  intarray *input_index = intarray_zeros(4);
  intarray *kernel_index = intarray_zeros(4);
  intarray *output_index = intarray_zeros(4);
  for (int b = 0; b < batch_size; b++) {
    input_index->items[0] = b;
    output_index->items[0] = b;
    for (int co = 0; co < cout; co++) {
      kernel_index->items[0] = co;
      output_index->items[1] = co;
      for (int h = 0; h < hin - kernel_size + 1; h++) {
        output_index->items[2] = h;
        for (int w = 0; w < win - kernel_size + 1; w++) {
          output_index->items[3] = w;
          float sum = 0;
          for (int ci = 0; ci < cin; ci++) {
            kernel_index->items[1] = ci;
            input_index->items[1] = ci;
            for (int k = 0; k < kernel_size * kernel_size; k++) {
              int kh = k / kernel_size;
              int kw = k % kernel_size;
              input_index->items[2] = h + kh;
              kernel_index->items[2] = kh;

              input_index->items[3] = w + kw;
              kernel_index->items[3] = kw;

              float kernel_value = tensor_getindex(kernels, kernel_index);
              float input_value = tensor_getindex(input, input_index);
              sum += input_value * kernel_value;
            }
          }
          storage_setindex(data, vw, output_index, sum);
        }
      }
    }
  }
  intarray_free(input_index);
  intarray_free(output_index);
  intarray_free(kernel_index);
  intarray_free(out_shape);
  return setup_tensor(vw, data, requires_grad, parents, CONV_2D,
                      _conv2d_backwards);
}

void _square_backwards(tensor *self) {
  tensor *twos = tensor_fill(self->vw->shape, 2, false);
  tensor *grads = tensor_mul(twos, self->parents[0], false);
  tensor *mul_self_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads =
      tensor_add(mul_self_grads, self->parents[0]->grads, false);
  tensor_free(self->parents[0]->grads);
  tensor_free(twos);
  tensor_free(grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_square(tensor *input, bool track_grads) {
  tensor **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(SQUARE) * sizeof(tensor *));
    parents[0] = input;
  }

  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < data->size; i++) {
    float value = storage_getitem(input->data, i);
    storage_setitem(data, i, value * value);
  }
  return setup_tensor(vw, data, requires_grad, parents, SQUARE,
                      _square_backwards);
}

void _sqrt_backwards(tensor *self) {
  tensor *copy_input = tensor_copy(self->parents[0], false);

  for (int i = 0; i < copy_input->data->size; i++) {
    float value = pow(storage_getitem(copy_input->data, i), -0.5f);
    storage_setitem(copy_input->data, i, value);
  }
  tensor *halfs = tensor_fill(self->vw->shape, 1.0 / 2, false);
  tensor *grads = tensor_mul(halfs, copy_input, false);
  tensor *mul_self_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads =
      tensor_add(mul_self_grads, self->parents[0]->grads, false);
  tensor_free(copy_input);
  tensor_free(self->parents[0]->grads);
  tensor_free(halfs);
  tensor_free(grads);
  tensor_free(mul_self_grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_sqrt(tensor *input, bool track_grads) {
  tensor **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(SQRT) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = sqrtf(storage_getitem(input->data, i));
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, SQRT, _sqrt_backwards);
}

void _exp_backwards(tensor *self) {
  tensor *mul = tensor_mul(self, self->grads, false);
  tensor *acc_grads = tensor_add(mul, self->parents[0]->grads, false);
  tensor_free(self->parents[0]->grads);
  tensor_free(mul);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_exp(tensor *input, bool track_grads) {
  tensor **parents = NULL;
  bool requires_grad = input->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(EXP) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = exp(storage_getitem(input->data, i));
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, EXP, _exp_backwards);
}

void _log_backwards(tensor *self) {
  tensor *copy_input = tensor_zeros(self->parents[0]->vw->shape, false);
  for (int i = 0; i < copy_input->data->size; i++) {
    float value = 1.0f / storage_getitem(copy_input->data, i);
    storage_setitem(copy_input->data, i, value);
  }
  tensor *mul = tensor_mul(copy_input, self->grads, false);
  tensor *acc_grads = tensor_add(mul, self->parents[0]->grads, false);
  tensor_free(self->parents[0]->grads);
  tensor_free(mul);
  tensor_free(copy_input);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_log(tensor *input, bool track_grads) {
  tensor **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(LOG) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = logf(storage_getitem(input->data, i));
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, LOG, _log_backwards);
}

void _reciprocal_backwards(tensor *self) {
  if (self->parents[0]->requires_grad) {
    // - 1 / x^2
    tensor *grads_neg = tensor_neg(self->grads, false);
    tensor *grads_mul = tensor_mul(grads_neg, self, false);
    tensor *grads_square = tensor_mul(grads_mul, self, false);
    tensor *acc_grads_1 =
        tensor_add(grads_square, self->parents[0]->grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(grads_neg);
    tensor_free(grads_mul);
    tensor_free(grads_square);
    self->parents[0]->grads = acc_grads_1;
  }
}

tensor *tensor_reciprocal(tensor *input, bool track_grads) {
  bool requires_grad = track_grads && input->requires_grad;
  tensor **parents = NULL;
  if (requires_grad) {
    parents =
        (tensor **)malloc(tensor_op_operands(RECIPROCAL) * sizeof(tensor *));
    parents[0] = input;
  }
  storage *data = storage_zeros(input->data->size);
  view *vw = view_copy(input->vw);
  for (uint64_t i = 0; i < input->data->size; i++) {
    float value = 1.0f / storage_getitem(input->data, i);
    storage_setitem(data, i, value);
  }
  return setup_tensor(vw, data, requires_grad, parents, RECIPROCAL,
                      _reciprocal_backwards);
}
