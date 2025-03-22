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

// TODO:
// rename a/s in function parameters to original_tensor, shape, etc.
// double check backwards functions are accumulating gradients and not just
// reseting them

storage *storage_new(uint64_t buffer_length) {
  float *buffer = (float *)calloc(buffer_length, sizeof(float));
  storage *s = (storage *)malloc(sizeof(storage));
  s->buffer = buffer;
  s->refcount = 1;
  s->size = buffer_length;
  return s;
}

// maybe not a good idea memory wise but whatever
storage *storage_from_buffer(uint64_t size, float *buffer) {
  // TODO: check if this works
  //
  // uint64_t size = malloc_usable_size(buffer)/sizeof(float);
  float *buffer_copy = (float *)calloc(size, sizeof(float)); // memcpy
  for (int i = 0; i < size; i++) {
    buffer_copy[i] = buffer[i];
  }
  storage *data = (storage *)malloc(sizeof(storage));
  data->size = size;
  data->buffer = buffer_copy;
  data->refcount = 1;
  return data;
}

void storage_free(storage *s) {
  free(s->buffer);
  free(s);
}

float storage_getitem(storage *s, uint64_t index) {
  assert(index >= 0 && index < s->size);
  return s->buffer[index];
}

void storage_setitem(storage *s, uint64_t index, float val) {
  assert(index >= 0 && index < s->size);
  s->buffer[index] = val;
}

void storage_inc_refcount(storage *s) { s->refcount++; }

void storage_dec_refcount(storage *s) {
  s->refcount--;
  if (s->refcount <= 0) {
    storage_free(s);
  }
}

storage *storage_copy(storage *s) {
  return storage_from_buffer(s->size, s->buffer);
}

void storage_to_zeros(storage *s) {
  free(s->buffer);
  s->buffer = (float *)calloc(s->size, sizeof(float));
}

// TODO: test please
uint64_t storage_logical_to_physical(tensor *t, intarray *logical_index) {
  intarray *t_strides = t->dview->strides;
  assert(logical_index->size == t->data->size);
  assert(logical_index->size == t_strides->size);

  uint64_t index = 0;
  for (int i = 0; i < logical_index->size; i++) {
    index += logical_index->items[i] * t_strides->items[i];
  }
  return index + t->dview->offset;
}

void view_free(view *view) {
  intarray_free(view->shape);
  intarray_free(view->strides);
  free(view);
}

tensor *tensor_zeros(intarray *s, bool requires_grad) {
  uint64_t size = intarray_prod(s);
  assert(size != 0);

  intarray *copy = intarray_copy(s);
  storage *data = storage_new(size);

  tensor *grads = NULL;
  if (requires_grad) {
    grads = tensor_zeros(s, false);
  }

  tensor *t = (tensor *)malloc(sizeof(tensor));

  // TODO: Make functions for views
  view *v = (view *)malloc(sizeof(view));
  t->dview = v;
  t->dview->shape = copy;
  t->dview->strides = intarray_ones(copy->size);
  t->dview->offset = 0;

  t->data = data;
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;
  return t;
}

tensor *tensor_ones(intarray *s, bool requires_grad) {
  tensor *ones = tensor_zeros(s, requires_grad);
  for (size_t i = 0; i < ones->data->size; i++) {
    storage_setitem(ones->data, i, 1.0f);
  }
  return ones;
}

tensor *tensor_from_buffer(intarray *s, float *buffer, bool requires_grad) {
  uint64_t size = intarray_prod(s);
  storage *data = storage_from_buffer(size, buffer);

  tensor *ret = (tensor *)malloc(sizeof(tensor));
  intarray *copy = intarray_copy(s);
  intarray *strides = intarray_ones(copy->size);

  view *v = (view *)malloc(sizeof(view));
  ret->dview = v;

  ret->dview->shape = copy;
  ret->dview->strides = strides;
  ret->dview->offset = 0;

  ret->data = data;

  tensor *grads = NULL;
  if (requires_grad) {
    grads = tensor_zeros(s, false);
  }
  ret->op = NOOP;
  ret->parents = NULL;
  ret->requires_grad = requires_grad;
  ret->_backwards = NULL;
  ret->grads = grads;
  return ret;
}

// TODO: optimization: pad once, then unpad.
// so shouldn't be doing pad here.
float tensor_getindex(tensor *self, intarray *index) {
  assert(index->size >= self->dview->shape->size);

  // TODO: maybe just copy self->dview->shape and get rid of if
  bool need_to_free = self->dview->shape->size < index->size;
  intarray *self_shape =
      self->dview->shape->size < index->size
          ? intarray_pad_left(self->dview->shape, index->size)
          : self->dview->shape;

  uint64_t index_buffer = 0;
  for (int i = 0; i < index->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < index->size; j++) {
      mul *= self_shape->items[j];
    }
    index_buffer += mul * index->items[i];
  }
  assert(index_buffer < intarray_prod(self_shape));
  if (need_to_free) {
    intarray_free(self_shape);
  }
  return self->data->buffer[index_buffer];
}

void tensor_setindex(tensor *self, intarray *index, float num) {
  intarray *self_shape =
      self->dview->shape->size < index->size
          ? intarray_pad_left(self->dview->shape, index->size)
          : self->dview->shape;
  bool need_to_free = self->dview->shape->size < index->size;
  uint64_t index_buffer = 0;
  for (int i = 0; i < index->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < index->size; j++) {
      mul *= self_shape->items[j];
    }
    index_buffer += mul * index->items[i];
  }
  assert(index_buffer < intarray_prod(self->dview->shape));
  if (need_to_free) {
    intarray_free(self_shape);
  }
  self->data->buffer[index_buffer] = num;
}

tensor *tensor_fill(intarray *s, float fill_value, bool requires_grad) {
  tensor *t = tensor_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    storage_setitem(t->data, i, fill_value);
  }
  return t;
}

tensor *tensor_linspace(intarray *s, float min, float max, bool requires_grad) {
  int steps = intarray_prod(s);
  tensor *t = tensor_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = min + i * ((max - min) / (steps - 1));
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_uniform(intarray *s, float min, float max, bool requires_grad) {
  tensor *t = tensor_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (float)rand() / (float)RAND_MAX * (max - min) + min;
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_uniformint(intarray *s, float min, float max, bool requires_grad) {
  tensor *t = tensor_uniform(s, min, max, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = round(storage_getitem(t->data, i));
    storage_setitem(t->data, i, value);
  }
  return t;
}

tensor *tensor_copy(tensor *original, bool requires_grad) {
  intarray *shape = intarray_copy(original->dview->shape);
  intarray *strides = intarray_copy(original->dview->strides);

  tensor *grads = NULL;
  if (requires_grad) {
    grads = tensor_zeros(shape, false);
  }

  tensor *t = (tensor *)malloc(sizeof(tensor));

  view *v = (view *)malloc(sizeof(view));
  t->dview = v;
  t->dview->shape = shape;
  t->dview->strides = strides;
  t->dview->offset = 0;

  t->data = storage_copy(original->data);
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;

  return t;
}

// where does this get used? 
// what happens if you call this during backprop?
void tensor_to_zeros(tensor *t) { storage_to_zeros(t->data); }

// same here
void tensor_to_n(struct tensor *t, float n) {
  for (int i = 0; i < t->data->size; i++) {
    storage_setitem(t->data, i, n);
  }
}

void tensor_print(tensor *t, bool show_buffer, bool show_grads) {
  if (!t) {
    printf("values: (null)\n");
    return;
  }
  intarray_print(t->dview->shape);
  if (t->requires_grad) {
    print_tensor_op(t->op);
  }
  if (show_buffer) {
    printf("values: [ ");
    for (int i = 0; i < t->data->size; i++) {
      printf("%f, ", t->data->buffer[i]);
    }
    printf("]\n");
  }
  if (t->requires_grad && show_grads) {
    printf("gradient shape: ");
    intarray_print(t->grads->dview->shape);
    printf("gradient values: [ ");
    for (int i = 0; i < t->grads->data->size; i++) {
      printf("%f, ", t->grads->data->buffer[i]);
    }
    printf("]\n");
  }
}

// should probably free any grads from children.
void tensor_free(tensor *t) {
  view_free(t->dview);
  storage_dec_refcount(t->data);

  free(t->parents);
  if (t->requires_grad) {
    tensor_free(t->grads); // make sure grads cant have grads
  }
  free(t);
}

bool tensor_equal(tensor *a, tensor *b, float rtol, float atol) {
  assert(intarray_equal(a->dview->shape, b->dview->shape));
  for (int i = 0; i < intarray_prod(a->dview->shape); i++) {
    float b_val = b->data->buffer[i];
    if (fabs(a->data->buffer[i] - b_val) > atol + rtol * fabs(b_val)) {
      return false;
    }
  }
  return true;
}

tensor *tensor_linear_init(intarray *shape, int in_features, bool requires_grad) {
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
  assert(intarray_equal(a->dview->shape, b->dview->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->dview->shape); // TODO: free copy!!

  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(ADD) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }

  tensor *t = tensor_zeros(copy, requires_grad);
  intarray_free(copy);
  t->parents = parents;
  t->op = ADD;
  t->_backwards = &_add_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] + b->data->buffer[i];
  }
  return t;
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
    self->parents[1]->grads = grads_1; // TODO: in place?
  }
}

tensor *tensor_sub(tensor *a, tensor *b, bool track_grads) {
  assert(intarray_equal(a->dview->shape, b->dview->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->dview->shape); // TODO: free copy??

  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(SUB) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }

  tensor *t = tensor_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = SUB;
  t->_backwards = &_sub_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] - b->data->buffer[i];
  }
  return t;
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
  assert(intarray_equal(a->dview->shape, b->dview->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->dview->shape);

  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(MUL) * sizeof(tensor *));
    parents[0] = a;
    parents[1] = b;
  }

  tensor *t = tensor_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = MUL;
  t->_backwards = &_mul_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * b->data->buffer[i];
  }
  return t;
}

void _sum_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  intarray *unit_shape = intarray_build(1, 1);

  intarray *self_shape = self->dview->shape;
  intarray *parent_shape = self->parents[0]->dview->shape;
  if (intarray_equal(unit_shape, self_shape)) {
    tensor *expanded_grads =
        tensor_fill(parent_shape, self->grads->data->buffer[0], false);
    tensor *acc_grads = tensor_add(self->parents[0]->grads, expanded_grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  } else {
    int expand_axis = 0;
    assert(self_shape->size == parent_shape->size);

    // TODO: i don't think this works if one of the dimensions was always 1.
    // make sure to check, especially if bs=1
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

    tensor *acc_grads = tensor_add(self->parents[0]->grads, expanded_grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  }
  intarray_free(unit_shape);
}

// axis=-1 => sum up all elements
// currently always keepdims, except for axis=-1
// could seriously use some tests here
tensor *tensor_sum(tensor *a, int axis, bool track_grads) {
  assert(axis >= -1 && axis < (int)a->dview->shape->size);
  intarray *new_shape;
  if (axis == -1) {
    new_shape = intarray_build(1, 1);
  } else {
    new_shape = intarray_copy(a->dview->shape);
    new_shape->items[axis] = 1;
  }

  bool requires_grad = a->requires_grad && track_grads;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(SUM_REDUCE) * sizeof(tensor *));
    parents[0] = a;
  }

  tensor *t = tensor_zeros(new_shape, requires_grad);
  t->parents = parents;
  t->op = SUM_REDUCE;
  t->_backwards = &_sum_backwards;

  if (axis == -1) {
    double sum = 0.0f;
    for (uint64_t i = 0; i < a->data->size; i++) {
      sum += a->data->buffer[i];
    }
    t->data->buffer[0] = sum;
  } else {
    intarray *stride = intarray_zeros(a->dview->shape->size);
    stride->items[axis] = 1;

    uint64_t along_axis = a->dview->shape->items[axis];
    uint64_t num_accumulate = intarray_prod(a->dview->shape) / along_axis;
    intarray *current = intarray_zeros(a->dview->shape->size);
    for (uint64_t i = 0; i < num_accumulate; i++) {
      float sum = 0.0f;
      for (uint64_t j = 0; j < along_axis; j++) {
        sum += tensor_getindex(a, current);
        current->items[axis]++;
      }
      current->items[axis] = 0;
      tensor_setindex(t, current, sum);
      // this looks kinda fucked but i think it works
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == axis)
          continue;
        current->items[k]++;
        if (current->items[k] >= a->dview->shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }
  }
  return t;
}

void _relu_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }

  tensor *grads = tensor_zeros(self->dview->shape, false);
  for (size_t i = 0; i < self->parents[0]->data->size; i++) {
    if (self->parents[0]->data->buffer[i] > 0) {
      grads->data->buffer[i] = 1;
    }
  }
  tensor *mul_grads = tensor_mul(self->grads, grads, false);
  tensor *acc_grads = tensor_add(self->parents[0]->grads, mul_grads, false);
  tensor_free(grads);
  tensor_free(self->parents[0]->grads);
  tensor_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_relu(tensor *a, bool track_grads) {
  intarray *copy = intarray_copy(a->dview->shape);
  tensor **parents = NULL;

  bool requires_grad = a->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(RELU) * sizeof(tensor *));
    parents[0] = a;
  }

  tensor *t = tensor_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = RELU;
  t->_backwards = &_relu_backwards;

  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * (a->data->buffer[i] > 0);
  }

  return t;
}

// Reshape
void _reshape_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad)
    return;
  tensor *grads = tensor_reshape(self->grads, self->parents[0]->dview->shape, false);
  tensor *acc_grads = tensor_add(grads, self->parents[0]->grads, false);
  tensor_free(grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_reshape(tensor *a, intarray *new_shape, bool track_grads) {
  intarray *new_shape_copy = intarray_copy(new_shape);
  assert(intarray_prod(new_shape) == intarray_prod(a->dview->shape));
  tensor **parents = NULL;
  tensor *reshaped_grads = NULL;
  bool requires_grad = a->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(RESHAPE) * sizeof(tensor *));
    parents[0] = a;
    reshaped_grads = tensor_reshape(a->grads, new_shape_copy, false);
  }
  tensor *t = tensor_copy(a, requires_grad);
  free(t->grads);
  t->dview->shape = new_shape_copy;
  t->parents = parents;
  t->op = RESHAPE;
  t->_backwards = &_reshape_backwards;
  t->grads = reshaped_grads;
  return t;
}

// Expand
// basically forwards sum (could totally do a refactor)
void _expand_backwards(tensor *self) {
  // sum
  // shape should be same for tensor and their gradients
  intarray *div =
      intarray_div(self->dview->shape, self->parents[0]->dview->shape);
  int expanded_axis = -1;
  for (int i = 0; i < div->size; i++) {
    if (div->items[i] != 1) {
      expanded_axis = i;
      break;
    }
  }
  assert(expanded_axis != -1 &&
         "Did not find an expanded axis from self->dview->shape");

  // sum self->parents[0]->grads along expanded_axis

  uint64_t along_axis = self->dview->shape->items[expanded_axis];
  uint64_t num_accumulate = intarray_prod(self->dview->shape) / along_axis;
  intarray *current = intarray_zeros(self->dview->shape->size);

  tensor *self_grad = self->grads;
  tensor *parent_grad = self->parents[0]->grads;
  for (uint64_t i = 0; i < num_accumulate; i++) {
    float sum = 0.0f;
    for (uint64_t j = 0; j < along_axis; j++) {
      sum += tensor_getindex(self_grad, current);
      current->items[expanded_axis]++;
    }
    current->items[expanded_axis] = 0;
    // TODO: bug, should be adding not setensoring
    tensor_setindex(parent_grad, current, sum);
    for (int k = current->size - 1; k >= 0; k--) {
      if (k == expanded_axis)
        continue;
      current->items[k]++;
      if (current->items[k] >= self->dview->shape->items[k]) {
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
tensor *tensor_expand(tensor *original_tensor, uint64_t axis, uint64_t factor,
              bool track_grads) {
  intarray *new_shape = intarray_copy(original_tensor->dview->shape);
  assert(axis >= 0 && axis < new_shape->size &&
         "Axis to expand is out of range.");
  assert(factor > 0 && "Expanding factor must be greater than 0");
  assert(new_shape->items[axis] == 1 && "Cannot expand [axis]!=1");

  // calculate new shape here
  new_shape->items[axis] *= factor; // TODO: check overflows.

  bool requires_grad = track_grads && original_tensor->requires_grad;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(EXPAND) * sizeof(tensor *));
    parents[0] = original_tensor;
  }

  tensor *expanded_tensor = tensor_zeros(new_shape, requires_grad);
  expanded_tensor->parents = parents;
  expanded_tensor->op = EXPAND;
  expanded_tensor->_backwards = &_expand_backwards;

  // expand here
  intarray *expanded_index = intarray_zeros(expanded_tensor->dview->shape->size);
  uint64_t along_axis = new_shape->items[axis];

  for (uint64_t i = 0; i < expanded_tensor->data->size; i++) {
    // expanding (like _sum_backwards)
    for (uint64_t j = 0; j < along_axis; j++) {
      intarray *original_index = intarray_copy(expanded_index);
      original_index->items[axis] = 0;
      float num = tensor_getindex(original_tensor, original_index);
      tensor_setindex(expanded_tensor, expanded_index, num);
      expanded_index->items[axis]++;
      intarray_free(original_index);
    }
    expanded_index->items[axis] = 0;
    // updating current (with expanded axis set to 0)
    for (int k = expanded_index->size - 1; k >= 0; k--) {
      if (k == axis) {
        continue;
      }
      expanded_index->items[k]++;
      if (expanded_index->items[k] >= original_tensor->dview->shape->items[k]) {
        expanded_index->items[k] = 0;
        continue;
      }
      break;
    }
  }
  return expanded_tensor;
}

void _neg_backwards(tensor *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tensor *grads = tensor_fill(self->dview->shape, -1.0f, false);
  tensor *mul_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads = tensor_add(mul_grads, self->parents[0]->grads, false);
  tensor_free(self->parents[0]->grads);
  tensor_free(grads);
  tensor_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tensor *tensor_neg(tensor *a, bool track_grads) {
  intarray *shape = intarray_copy(a->dview->shape);

  bool requires_grad = track_grads && a->requires_grad;
  tensor *t = tensor_zeros(shape, requires_grad);
  tensor **parents = NULL;
  if (track_grads) {
    parents = (tensor **)malloc(tensor_op_operands(NEG) * sizeof(tensor *));
    parents[0] = a;
  }

  t->parents = parents;
  t->op = NEG;
  t->_backwards = &_neg_backwards;

  for (uint64_t i = 0; i < a->data->size; i++) {
    float value = storage_getitem(a->data, i);
    storage_setitem(t->data, i, -value);
  }
  return t;
}

// still assuming square kernel
void _maxpool2d_backwards(tensor *self) {
  // parent grads will be tensor, only with 1s or 0s
  // so basically max pool, except keep track of max index.
  // set max index to 1, others to 0
  // mul expanded by sparse grads
  // then acc to parents->grads.

  intarray *self_shape = self->dview->shape;
  intarray *parent_shape = self->parents[0]->dview->shape;
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
  tensor *pooled_grads = tensor_ones(self->parents[0]->dview->shape, false);
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
            float value =
                tensor_getindex(self->parents[0],
                            index); // backprop is dependent on tensors staying
            if (value > max) {      // if equal, its the first one.
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
  intarray *input_shape = input->dview->shape;
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
    parents = (tensor **)malloc(tensor_op_operands(MAX_POOL_2D) * sizeof(tensor *));
    parents[0] = input;
  }
  tensor *output = tensor_zeros(new_shape, requires_grad);
  output->parents = parents;
  output->op = MAX_POOL_2D;
  output->_backwards = &_maxpool2d_backwards;

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
          tensor_setindex(output, index, max);
        }
      }
    }
  }
  intarray_free(index);
  return output;
}

void _matmul_backwards(tensor *self) {
  // weights: (transpose inputs * self.grad) sum
  if (self->parents[0]->requires_grad) {
  }
  // inputs: (transpose weights * self.grad) expand
  if (self->parents[1]->requires_grad) {
  }
}

// for now, weights will be 2d
// input will be 2/3d (for batches)
tensor *tensor_matmul(tensor *w, tensor *b, bool track_grads) {
  int weights_size = w->dview->shape->size;
  int batch_size = b->dview->shape->size;
  assert(weights_size == 2);
  assert(batch_size == 2 || batch_size == 3);

  int ww = w->dview->shape->items[weights_size - 1];
  int wh = w->dview->shape->items[weights_size - 2];

  int bw = b->dview->shape->items[batch_size - 1];
  int bh = b->dview->shape->items[batch_size - 2];

  int bs = batch_size == 3 ? b->dview->shape->items[0] : 1;

  assert(ww == bh && "Tensors are not the correct shape");

  intarray *new_shape;
  if (batch_size == 3) {
    new_shape = intarray_build(3, bs, wh, bw);
  } else {
    new_shape = intarray_build(2, wh, bw);
  }

  tensor **parents = NULL;
  bool requires_grad = (w->requires_grad || b->requires_grad) && track_grads;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(MATMUL) * sizeof(tensor *));
    parents[0] = w;
    parents[1] = b;
  }

  tensor *t = tensor_zeros(new_shape, requires_grad);
  intarray_free(new_shape);
  t->parents = parents;
  t->op = MATMUL;
  t->_backwards = &_matmul_backwards;

  intarray *wi = intarray_zeros(2);
  intarray *bi = intarray_zeros(3);
  intarray *oi = intarray_zeros(3);

  for (int batch = 0; batch < bs; batch++) {
    oi->items[0] = batch;
    bi->items[0] = batch;
    for (int k = 0; k < wh; k++) {
      oi->items[1] = k;
      wi->items[0] = k;
      for (int j = 0; j < bw; j++) {
        oi->items[2] = j;
        bi->items[2] = j;
        float sum = 0;
        for (int i = 0; i < ww; i++) {
          bi->items[1] = i;
          wi->items[1] = i;
          float av = tensor_getindex(w, wi);
          float bv = tensor_getindex(b, bi);
          sum += av * bv;
        }
        tensor_setindex(t, oi, sum);
      }
    }
  }
  intarray_free(wi);
  intarray_free(bi);
  intarray_free(oi);
  return t;
}

void _conv2d_backwards(tensor *self) {
  int batch_size = self->dview->shape->items[0];
  int cout = self->dview->shape->items[1];
  int cin = self->parents[0]->dview->shape->items[1];
  int win = self->parents[0]->dview->shape->items[3];
  int hin = self->parents[0]->dview->shape->items[2];
  int kernel_size = self->parents[1]->dview->shape->items[3];

  // input gradients
  // TODO: refactor into one
  if (self->parents[0]->requires_grad) {
    tensor *grads = tensor_zeros(self->parents[0]->dview->shape, false);
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
    tensor *grads = tensor_zeros(self->parents[1]->dview->shape, false);
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

                float input_value = tensor_getindex(self->parents[0], input_index);
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
  intarray *input_shape = input->dview->shape;
  assert(input_shape->size == 4);

  intarray *kernels_shape = kernels->dview->shape;
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

  tensor *output = tensor_zeros(out_shape, requires_grad);
  output->parents = parents;
  output->op = CONV_2D;
  output->_backwards = &_conv2d_backwards;

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
          tensor_setindex(output, output_index, sum);
        }
      }
    }
  }
  intarray_free(input_index);
  intarray_free(output_index);
  intarray_free(kernel_index);
  return output;
}

void _square_backwards(tensor *self) {
  tensor *twos = tensor_fill(self->dview->shape, 2, false);
  tensor *grads = tensor_mul(twos, self->parents[0], false);
  tensor *mul_self_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads = tensor_add(mul_self_grads, self->parents[0]->grads, false);
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

  tensor *t = tensor_zeros(input->dview->shape, requires_grad);
  t->parents = parents;
  t->op = SQUARE;
  t->_backwards = &_square_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = pow(input->data->buffer[i], 2);
  }
  return t;
}

void _sqrt_backwards(tensor *self) {
  tensor *copy_input = tensor_copy(self->parents[0], false);

  for (int i = 0; i < copy_input->data->size; i++) {
    copy_input->data->buffer[i] = pow(copy_input->data->buffer[i], -.5);
  }
  tensor *halfs = tensor_fill(self->dview->shape, 1.0 / 2, false);
  tensor *grads = tensor_mul(halfs, copy_input, false);
  tensor *mul_self_grads = tensor_mul(grads, self->grads, false);
  tensor *acc_grads = tensor_add(mul_self_grads, self->parents[0]->grads, false);
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
  tensor *t = tensor_zeros(input->dview->shape, requires_grad);
  t->parents = parents;
  t->op = SQRT;
  t->_backwards = &_sqrt_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = sqrtf(input->data->buffer[i]);
  }
  return t;
}

// TODO: inplace instead?
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
  tensor *t = tensor_zeros(input->dview->shape, requires_grad);
  t->parents = parents;
  t->op = EXP;
  t->_backwards = &_exp_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = exp(input->data->buffer[i]);
  }
  return t;
}

void _log_backwards(tensor *self) {
  tensor *copy_input = tensor_zeros(self->parents[0]->dview->shape, false);
  for (int i = 0; i < copy_input->data->size; i++) {
    copy_input->data->buffer[i] = 1.0f / self->parents[0]->data->buffer[i];
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
  tensor *t = tensor_zeros(input->dview->shape, requires_grad);
  t->parents = parents;
  t->op = LOG;
  t->_backwards = &_log_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = logf(input->data->buffer[i]);
  }
  return t;
}

void _reciprocal_backwards(tensor *self) {
  if (self->parents[0]->requires_grad) {
    // - 1 / x^2
    tensor *grads_neg = tensor_neg(self->grads, false);
    tensor *grads_mul = tensor_mul(grads_neg, self, false);
    tensor *grads_square = tensor_mul(grads_mul, self, false);
    tensor *acc_grads_1 = tensor_add(grads_square, self->parents[0]->grads, false);
    tensor_free(self->parents[0]->grads);
    tensor_free(grads_neg);
    tensor_free(grads_mul);
    tensor_free(grads_square);
    self->parents[0]->grads = acc_grads_1;
  }
}

tensor *tensor_reciprocal(tensor *a, bool track_grads) {
  intarray *copy = intarray_copy(a->dview->shape);

  bool requires_grad = track_grads && a->requires_grad;
  tensor **parents = NULL;
  if (requires_grad) {
    parents = (tensor **)malloc(tensor_op_operands(RECIPROCAL) * sizeof(tensor *));
    parents[0] = a;
  }

  tensor *t = tensor_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = RECIPROCAL;
  t->_backwards = &_reciprocal_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = 1.0f / a->data->buffer[i];
  }
  return t;
}
