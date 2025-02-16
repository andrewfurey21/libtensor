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

// TODO:
// rename a/s in function parameters to original_tensor, shape, etc.
// double check backwards functions are accumulating gradients and not just
// reseting them

tstorage *tstorage_new(uint64_t buffer_length) {
  float *buffer = (float *)calloc(buffer_length, sizeof(float));
  tstorage *storage = (tstorage *)malloc(sizeof(tstorage));
  storage->buffer = buffer;
  storage->refcount = 1;
  storage->size = buffer_length;
  return storage;
}

// maybe not a good idea memory wise but whatever
tstorage *tstorage_from_buffer(uint64_t size, float *buffer) {
  // TODO: check if this works
  //
  // uint64_t size = malloc_usable_size(buffer)/sizeof(float);
  float *buffer_copy = (float *)calloc(size, sizeof(float)); // memcpy
  for (int i = 0; i < size; i++) {
    buffer_copy[i] = buffer[i];
  }
  tstorage *data = (tstorage *)malloc(sizeof(tstorage));
  data->size = size;
  data->buffer = buffer_copy;
  data->refcount = 1;
  return data;
}

void tstorage_free(tstorage *s) {
  free(s->buffer);
  free(s);
}

float tstorage_getitem(tstorage *s, uint64_t index) {
  assert(index >= 0 && index < s->size);
  return s->buffer[index];
}

void tstorage_setitem(tstorage *s, uint64_t index, float val) {
  assert(index >= 0 && index < s->size);
  s->buffer[index] = val;
}

void tstorage_inc_refcount(tstorage *s) { s->refcount++; }

void tstorage_dec_refcount(tstorage *s) {
  s->refcount--;
  if (s->refcount <= 0) {
    tstorage_free(s);
  }
}

tstorage *tstorage_copy(tstorage *s) {
  return tstorage_from_buffer(s->size, s->buffer);
}

void tstorage_to_zeros(tstorage *s) {
  free(s->buffer);
  s->buffer = (float *)calloc(s->size, sizeof(float));
}

// TODO: test please
uint64_t tstorage_logical_to_physical(tt *t, intarray *logical_index) {
  intarray *t_strides = t->view->strides;
  assert(logical_index->size == t->data->size);
  assert(logical_index->size == t_strides->size);

  uint64_t index = 0;
  for (int i = 0; i < logical_index->size; i++) {
    index += logical_index->items[i] * t_strides->items[i];
  }
  return index + t->view->offset;
}

void tview_free(tview *view) {
  intarray_free(view->shape);
  intarray_free(view->strides);
  free(view);
}

tt *tt_zeros(intarray *s, bool requires_grad) {
  uint64_t size = intarray_prod(s);
  assert(size != 0);

  intarray *copy = intarray_copy(s);
  tstorage *data = tstorage_new(size);

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(s, false);
  }

  tt *t = (tt *)malloc(sizeof(tt));

  // TODO: Make functions for views
  tview *view = (tview *)malloc(sizeof(tview));
  t->view = view;
  t->view->shape = copy;
  t->view->strides = intarray_ones(copy->size);
  t->view->offset = 0;

  t->data = data;
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;
  return t;
}

tt *tt_ones(intarray *s, bool requires_grad) {
  tt *ones = tt_zeros(s, requires_grad);
  for (size_t i = 0; i < ones->data->size; i++) {
    tstorage_setitem(ones->data, i, 1.0f);
  }
  return ones;
}

tt *tt_from_buffer(intarray *s, float *buffer, bool requires_grad) {
  uint64_t size = intarray_prod(s);
  tstorage *data = tstorage_from_buffer(size, buffer);

  tt *ret = (tt *)malloc(sizeof(tt));
  intarray *copy = intarray_copy(s);
  intarray *strides = intarray_ones(copy->size);

  tview *view = (tview *)malloc(sizeof(tview));
  ret->view = view;

  ret->view->shape = copy;
  ret->view->strides = strides;
  ret->view->offset = 0;

  ret->data = data;

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(s, false);
  }
  ret->op = NOOP;
  ret->parents = NULL;
  ret->requires_grad = requires_grad;
  ret->_backwards = NULL;
  ret->grads = grads;
  return ret;
}

float tt_getindex(tt *self, intarray *s) {
  intarray *self_shape = self->view->shape->size < s->size
                           ? intarray_add_one(self->view->shape)
                           : self->view->shape;
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  assert(index < intarray_prod(self_shape));
  return self->data->buffer[index];
}

void tt_setindex(tt *self, intarray *s, float num) {
  intarray *self_shape = self->view->shape->size < s->size
                           ? intarray_add_one(self->view->shape)
                           : self->view->shape;
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  assert(index < intarray_prod(self->view->shape));
  self->data->buffer[index] = num;
}

tt *tt_fill(intarray *s, float fill_value, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    tstorage_setitem(t->data, i, fill_value);
  }
  return t;
}

tt *tt_linspace(intarray *s, float min, float max, bool requires_grad) {
  int steps = intarray_prod(s);
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = min + i * ((max - min) / (steps - 1));
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniform(intarray *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (float)rand() / (float)RAND_MAX * (max - min) + min;
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniformint(intarray *s, float min, float max, bool requires_grad) {
  tt *t = tt_uniform(s, min, max, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = round(tstorage_getitem(t->data, i));
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_copy(tt *original, bool requires_grad) {
  intarray *shape = intarray_copy(original->view->shape);
  intarray *strides = intarray_copy(original->view->strides);

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(shape, false);
  }

  tt *t = (tt *)malloc(sizeof(tt));

  tview *view = (tview *)malloc(sizeof(tview));
  t->view = view;
  t->view->shape = shape;
  t->view->strides = strides;
  t->view->offset = 0;

  t->data = tstorage_copy(original->data);
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;

  return t;
}

void tt_to_zeros(tt *t) { tstorage_to_zeros(t->data); }

void tt_to_n(struct tt *t, float n) {
  for (int i = 0; i < t->data->size; i++) {
    tstorage_setitem(t->data, i, n);
  }
}

void tt_print(tt *t, bool show_buffer, bool show_grads) {
  if (!t) {
    printf("values: (null)\n");
    return;
  }
  intarray_print(t->view->shape);
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
    intarray_print(t->grads->view->shape);
    printf("gradient values: [ ");
    for (int i = 0; i < t->grads->data->size; i++) {
      printf("%f, ", t->grads->data->buffer[i]);
    }
    printf("]\n");
  }
}

// should probably free any grads from children.
void tt_free(tt *t) {
  tview_free(t->view);
  tstorage_dec_refcount(t->data);

  free(t->parents);
  if (t->requires_grad) {
    tt_free(t->grads); // make sure grads cant have grads
  }
  free(t);
}

// void tt_free_parents(tt *t) {
//   for (int i = 0; i < tensor_op_operands(t->op); i++) {
//     tt_free(t->parents[i]);
//   }
//   free(t->parents);
// }

bool tt_equal(tt *a, tt *b) {
  assert(intarray_equal(a->view->shape, b->view->shape));
  for (int i = 0; i < intarray_prod(a->view->shape); i++) {
    if (fabs(a->data->buffer[i] - b->data->buffer[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

tt *tt_linear_init(intarray *shape, int in_features, bool requires_grad) {
  float bound = 1.0f / sqrtf((float)in_features);
  return tt_uniform(shape, -bound, bound, requires_grad);
}

tt *tt_conv_init(intarray *shape, int in_channels, int kernel_size,
                 bool requires_grad) {
  float scale = 1.0f / sqrtf((float)(in_channels * kernel_size * kernel_size));
  return tt_uniform(shape, -scale, scale, requires_grad);
}

void _add_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads, false);
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_add(self->grads, self->parents[1]->grads, false);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}
// TODO: need to add track_gradients
// tt_add(tt* a, tt* b, bool track_gradients);
tt *tt_add(tt *a, tt *b, bool track_grads) {
  assert(intarray_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->view->shape); // TODO: free copy!!
  
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(ADD) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  intarray_free(copy);
  t->parents = parents;
  t->op = ADD;
  t->_backwards = &_add_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] + b->data->buffer[i];
  }
  return t;
}

void _sub_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads, false);
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_sub(self->parents[1]->grads, self->grads, false);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1; // TODO: in place?
  }
}

tt *tt_sub(tt *a, tt *b, bool track_grads) {
  assert(intarray_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->view->shape); // TODO: free copy??

  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(SUB) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = SUB;
  t->_backwards = &_sub_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] - b->data->buffer[i];
  }
  return t;
}

void _mul_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_mul(self->grads, self->parents[1], false);
    tt *acc_grads_0 = tt_add(grads_0, self->parents[0]->grads, false);
    tt_free(self->parents[0]->grads);
    tt_free(grads_0);
    self->parents[0]->grads = acc_grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_mul(self->grads, self->parents[0], false);
    tt *acc_grads_1 = tt_add(grads_1, self->parents[1]->grads, false);
    tt_free(self->parents[1]->grads);
    tt_free(grads_1);
    self->parents[1]->grads = acc_grads_1;
  }
}

tt *tt_mul(tt *a, tt *b, bool track_grads) {
  assert(intarray_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  intarray *copy = intarray_copy(a->view->shape);

  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(MUL) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = MUL;
  t->_backwards = &_mul_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * b->data->buffer[i];
  }
  return t;
}

void _sum_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  intarray *unit_shape = intarray_build(1, 1);

  intarray *self_shape = self->view->shape;
  intarray *parent_shape = self->parents[0]->view->shape;
  if (intarray_equal(unit_shape, self_shape)) {
    tt *expanded_grads =
        tt_fill(parent_shape, self->grads->data->buffer[0], false);
    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads, false);
    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);
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

    tt *expanded_grads = tt_zeros(parent_shape, false);

    intarray *current = intarray_zeros(parent_shape->size);
    uint64_t along_axis = parent_shape->items[expand_axis];
    for (uint64_t i = 0; i < self->grads->data->size; i++) {
      // expanding
      for (uint64_t j = 0; j < along_axis; j++) {
        intarray *current_grads = intarray_copy(current);
        current_grads->items[expand_axis] = 0;
        float num = tt_getindex(self->grads, current_grads);
        tt_setindex(expanded_grads, current, num);
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

    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads, false);
    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  }
  intarray_free(unit_shape);
}

// axis=-1 => sum up all elements
// currently always keepdims, except for axis=-1
// could seriously use some tests here
tt *tt_sum(tt *a, int axis, bool track_grads) {
  assert(axis >= -1 && axis < (int)a->view->shape->size);
  intarray *new_shape;
  if (axis == -1) {
    new_shape = intarray_build(1, 1);
  } else {
    new_shape = intarray_copy(a->view->shape);
    new_shape->items[axis] = 1;
  }

  bool requires_grad = a->requires_grad && track_grads;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(SUM_REDUCE) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(new_shape, requires_grad);
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
    intarray *stride = intarray_zeros(a->view->shape->size);
    stride->items[axis] = 1;

    uint64_t along_axis = a->view->shape->items[axis];
    uint64_t num_accumulate = intarray_prod(a->view->shape) / along_axis;
    intarray *current = intarray_zeros(a->view->shape->size);
    for (uint64_t i = 0; i < num_accumulate; i++) {
      float sum = 0.0f;
      for (uint64_t j = 0; j < along_axis; j++) {
        sum += tt_getindex(a, current);
        current->items[axis]++;
      }
      current->items[axis] = 0;
      tt_setindex(t, current, sum);
      // this looks kinda fucked but i think it works
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == axis)
          continue;
        current->items[k]++;
        if (current->items[k] >= a->view->shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }
  }
  return t;
}

void _relu_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }

  tt *grads = tt_zeros(self->view->shape, false);
  for (size_t i = 0; i < self->parents[0]->data->size; i++) {
    if (self->parents[0]->data->buffer[i] > 0) {
      grads->data->buffer[i] = 1;
    }
  }
  tt *mul_grads = tt_mul(self->grads, grads, false);
  tt *acc_grads = tt_add(self->parents[0]->grads, mul_grads, false);
  tt_free(grads);
  tt_free(self->parents[0]->grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_relu(tt *a, bool track_grads) {
  intarray *copy = intarray_copy(a->view->shape);
  tt **parents = NULL;

  bool requires_grad = a->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(RELU) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = RELU;
  t->_backwards = &_relu_backwards;

  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * (a->data->buffer[i] > 0);
  }

  return t;
}

// Reshape
void _reshape_backwards(tt *self) {
  if (!self->parents[0]->requires_grad)
    return;
  tt *grads = tt_reshape(self->grads, self->parents[0]->view->shape, false);
  tt *acc_grads = tt_add(grads, self->parents[0]->grads, false);
  tt_free(grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_reshape(tt *a, intarray *new_shape, bool track_grads) {
  intarray *new_shape_copy = intarray_copy(new_shape);
  assert(intarray_prod(new_shape) == intarray_prod(a->view->shape));
  tt **parents = NULL;
  tt *reshaped_grads = NULL;
  bool requires_grad = a->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(RESHAPE) * sizeof(tt *));
    parents[0] = a;
    reshaped_grads = tt_reshape(a->grads, new_shape_copy, false);
  }
  tt *t = tt_copy(a, requires_grad);
  free(t->grads);
  t->view->shape = new_shape_copy;
  t->parents = parents;
  t->op = RESHAPE;
  t->_backwards = &_reshape_backwards;
  t->grads = reshaped_grads;
  return t;
}

// Expand
// basically forwards sum (could totally do a refactor)
void _expand_backwards(tt *self) {
  // sum
  // shape should be same for tensor and their gradients
  intarray *div = intarray_div(self->view->shape, self->parents[0]->view->shape);
  int expanded_axis = -1;
  for (int i = 0; i < div->size; i++) {
    if (div->items[i] != 1) {
      expanded_axis = i;
      break;
    }
  }
  assert(expanded_axis != -1 &&
         "Did not find an expanded axis from self->view->shape");

  // sum self->parents[0]->grads along expanded_axis

  uint64_t along_axis = self->view->shape->items[expanded_axis];
  uint64_t num_accumulate = intarray_prod(self->view->shape) / along_axis;
  intarray *current = intarray_zeros(self->view->shape->size);

  tt *self_grad = self->grads;
  tt *parent_grad = self->parents[0]->grads;
  for (uint64_t i = 0; i < num_accumulate; i++) {
    float sum = 0.0f;
    for (uint64_t j = 0; j < along_axis; j++) {
      sum += tt_getindex(self_grad, current);
      current->items[expanded_axis]++;
    }
    current->items[expanded_axis] = 0;
    // TODO: bug, should be adding not setting
    tt_setindex(parent_grad, current, sum);
    for (int k = current->size - 1; k >= 0; k--) {
      if (k == expanded_axis)
        continue;
      current->items[k]++;
      if (current->items[k] >= self->view->shape->items[k]) {
        current->items[k] = 0;
        continue;
      }
      break;
    }
  }
}

// currently, must expand, cannot contract.
// can expand axis where dim>=1
// basically backwards sum
// follows broadcasting rules, cannot expand dim that isn't 1
tt *tt_expand(tt *original_tensor, uint64_t axis, uint64_t factor,
              bool track_grads) {
  intarray *new_shape = intarray_copy(original_tensor->view->shape);
  assert(axis >= 0 && axis < new_shape->size &&
         "Axis to expand is out of range.");
  assert(factor > 0 && "Expanding factor must be greater than 0");
  assert(new_shape->items[axis] == 1 && "Cannot expand [axis]!=1");

  // calculate new shape here
  new_shape->items[axis] *= factor; // TODO: check overflows.

  bool requires_grad = track_grads && original_tensor->requires_grad;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(EXPAND) * sizeof(tt *));
    parents[0] = original_tensor;
  }

  tt *expanded_tensor = tt_zeros(new_shape, requires_grad);
  expanded_tensor->parents = parents;
  expanded_tensor->op = EXPAND;
  expanded_tensor->_backwards = &_expand_backwards;

  // expand here
  intarray *expanded_index = intarray_zeros(expanded_tensor->view->shape->size);
  uint64_t along_axis = new_shape->items[axis];

  for (uint64_t i = 0; i < expanded_tensor->data->size; i++) {
    // expanding (like _sum_backwards)
    for (uint64_t j = 0; j < along_axis; j++) {
      intarray *original_index = intarray_copy(expanded_index);
      original_index->items[axis] = 0;
      float num = tt_getindex(original_tensor, original_index);
      tt_setindex(expanded_tensor, expanded_index, num);
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
      if (expanded_index->items[k] >= original_tensor->view->shape->items[k]) {
        expanded_index->items[k] = 0;
        continue;
      }
      break;
    }
  }
  return expanded_tensor;
}

void _neg_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tt *grads = tt_fill(self->view->shape, -1.0f, false);
  tt *mul_grads = tt_mul(grads, self->grads, false);
  tt *acc_grads = tt_add(mul_grads, self->parents[0]->grads, false);
  tt_free(self->parents[0]->grads);
  tt_free(grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_neg(tt *a, bool track_grads) {
  intarray *shape = intarray_copy(a->view->shape);

  bool requires_grad = track_grads && a->requires_grad;
  tt *t = tt_zeros(shape, requires_grad);
  tt **parents = NULL;
  if (track_grads) {
    parents = (tt **)malloc(tensor_op_operands(NEG) * sizeof(tt *));
    parents[0] = a;
  }

  t->parents = parents;
  t->op = NEG;
  t->_backwards = &_neg_backwards;

  for (uint64_t i = 0; i < a->data->size; i++) {
    float value = tstorage_getitem(a->data, i);
    tstorage_setitem(t->data, i, -value);
  }
  return t;
}

// still assuming square kernel
void _maxpool2d_backwards(tt *self) {
  // parent grads will be tensor, only with 1s or 0s
  // so basically max pool, except keep track of max index.
  // set max index to 1, others to 0
  // mul expanded by sparse grads
  // then acc to parents->grads.

  intarray *self_shape = self->view->shape;
  intarray *parent_shape = self->parents[0]->view->shape;
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
  tt *expanded_self_grad = tt_zeros(parent_shape, false);
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
          float value = tt_getindex(self->grads, index);

          index->items[x_index] = ow;
          index->items[x_index - 1] = oh;
          tt_setindex(expanded_self_grad, index, value);
        }
      }
    }
  }
  intarray_free(index);

  index = intarray_zeros(parent_shape->size);
  tt *pooled_grads = tt_ones(self->parents[0]->view->shape, false);
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
                tt_getindex(self->parents[0],
                            index); // backprop is dependent on tensors staying
            if (value > max) {      // if equal, its the first one.
              if (max != -INFINITY)
                tt_setindex(pooled_grads, max_index, 0);
              max = value;
              intarray_free(max_index);
              max_index = intarray_copy(index);
            } else {
              tt_setindex(pooled_grads, index, 0);
            }
          }
          intarray_free(max_index);
        }
      }
    }
  }
  intarray_free(index);

  tt *expanded_by_pooled_grads =
      tt_mul(expanded_self_grad, pooled_grads, false);
  tt *accumulated_grads =
      tt_add(self->parents[0]->grads, expanded_by_pooled_grads, false);
  tt_free(expanded_self_grad);
  tt_free(pooled_grads);
  self->parents[0]->grads = accumulated_grads;
}

// NOTE:
// assuming input is divisible by kernel size
// stride is kernel size
// no dilation, padding. ceilmode=False.
// 4d, 3d, 2d only.
tt *tt_maxpool2d(tt *input, int kernel_size, bool track_grads) {
  intarray *input_shape = input->view->shape;
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

  tt **parents = NULL;
  bool requires_grad = input->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(MAX_POOL_2D) * sizeof(tt *));
    parents[0] = input;
  }
  tt *output = tt_zeros(new_shape, requires_grad);
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
            float value = tt_getindex(input, index);
            if (value > max)
              max = value;
          }
          int x = ow / kernel_size;
          int y = oh / kernel_size;
          index->items[x_index - 1] = y;
          index->items[x_index] = x;
          tt_setindex(output, index, max);
        }
      }
    }
  }
  intarray_free(index);
  return output;
}

// broadcasting would get rid of the mallocs i think.
void _matmul_backwards(tt *self) {
  // weights: (transpose inputs * self.grad) sum
  if (self->parents[0]->requires_grad) {
  }
  // inputs: (transpose weights * self.grad) expand
  if (self->parents[1]->requires_grad) {
  }
}

// for now, a can have 2d
// b can have 2d or 3d (for batches)
// need to do broadcasting
tt *tt_matmul(tt *a, tt *b, bool track_grads) {
  int a_size = a->view->shape->size;
  int b_size = b->view->shape->size;
  assert(a_size == 2);
  assert(b_size == 2 || b_size == 3);

  int aw = a->view->shape->items[a_size - 1];
  int ah = a->view->shape->items[a_size - 2];

  int bw = b->view->shape->items[b_size - 1];
  int bh = b->view->shape->items[b_size - 2];

  int bs = b_size == 3 ? b->view->shape->items[0] : 1;

  assert(aw == bh && "Tensors are not the correct shape");

  intarray *new_shape;
  if (b_size == 3) {
    new_shape = intarray_build(3, bs, ah, bw);
  } else {
    new_shape = intarray_build(2, ah, bw);
  }

  tt **parents = NULL;
  bool requires_grad = (a->requires_grad || b->requires_grad) && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(MATMUL) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(new_shape, requires_grad);
  intarray_free(new_shape);
  t->parents = parents;
  t->op = MATMUL;
  t->_backwards = &_matmul_backwards;

  intarray *ai = intarray_zeros(2);
  intarray *bi = intarray_zeros(3);
  intarray *oi = intarray_zeros(3);

  for (int batch = 0; batch < bs; batch++) {
    oi->items[0] = batch;
    bi->items[0] = batch;
    for (int k = 0; k < ah; k++) {
      oi->items[1] = k;
      ai->items[0] = k;
      for (int j = 0; j < bw; j++) {
        oi->items[2] = j;
        bi->items[2] = j;
        float sum = 0;
        for (int i = 0; i < aw; i++) {
          bi->items[1] = i;
          ai->items[1] = i;
          float av = tt_getindex(a, ai);
          float bv = tt_getindex(b, bi);
          sum += av * bv;
        }
        tt_setindex(t, oi, sum);
      }
    }
  }
  intarray_free(ai);
  intarray_free(bi);
  intarray_free(oi);
  return t;
}

void _conv2d_backwards(tt *self) {
  int batch_size = self->view->shape->items[0];
  int cout = self->view->shape->items[1];
  int cin = self->parents[0]->view->shape->items[1];
  int win = self->parents[0]->view->shape->items[3];
  int hin = self->parents[0]->view->shape->items[2];
  int kernel_size = self->parents[1]->view->shape->items[3];

  // input gradients
  // TODO: refactor into one
  if (self->parents[0]->requires_grad) {
    tt *grads = tt_zeros(self->parents[0]->view->shape, false);
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

                float current_value = tt_getindex(grads, input_grad_index);
                float kernel_value =
                    tt_getindex(self->parents[1], kernel_index);
                float output_grad_value =
                    tt_getindex(self->grads, output_grad_index);
                float new_value =
                    kernel_value * output_grad_value + current_value;
                tt_setindex(grads, input_grad_index, new_value);
              }
            }
          }
        }
      }
    }
    tt *acc_grads = tt_add(grads, self->parents[0]->grads, false);

    tt_free(grads);
    tt_free(self->parents[0]->grads);

    intarray_free(kernel_index);
    intarray_free(input_grad_index);
    intarray_free(output_grad_index);

    self->parents[0]->grads = acc_grads;
  }

  // kernel gradients
  if (self->parents[1]->requires_grad) {
    tt *grads = tt_zeros(self->parents[1]->view->shape, false);
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

                float input_value = tt_getindex(self->parents[0], input_index);
                float current_value = tt_getindex(grads, kernel_grad_index);
                float output_grad_value =
                    tt_getindex(self->grads, output_grad_index);
                float new_value =
                    input_value * output_grad_value + current_value;
                tt_setindex(grads, kernel_grad_index, new_value);
              }
            }
          }
        }
      }
    }
    tt *acc_grads = tt_add(grads, self->parents[1]->grads, false);

    tt_free(grads);
    tt_free(self->parents[1]->grads);

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
tt *tt_conv2d(tt *input, tt *kernels, bool track_grads) {
  intarray *input_shape = input->view->shape;
  assert(input_shape->size == 4);

  intarray *kernels_shape = kernels->view->shape;
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

  tt **parents = NULL;
  bool requires_grad = (input->requires_grad || kernels->requires_grad) && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(CONV_2D) * sizeof(tt *));
    parents[0] = input;
    parents[1] = kernels;
  }

  tt *output = tt_zeros(out_shape, requires_grad);
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

              float kernel_value = tt_getindex(kernels, kernel_index);
              float input_value = tt_getindex(input, input_index);
              sum += input_value * kernel_value;
            }
          }
          tt_setindex(output, output_index, sum);
        }
      }
    }
  }
  intarray_free(input_index);
  intarray_free(output_index);
  intarray_free(kernel_index);
  return output;
}

void _square_backwards(tt *self) {
  tt *twos = tt_fill(self->view->shape, 2, false);
  tt *grads = tt_mul(twos, self->parents[0], false);
  tt *mul_self_grads = tt_mul(grads, self->grads, false);
  tt *acc_grads = tt_add(mul_self_grads, self->parents[0]->grads, false);
  tt_free(self->parents[0]->grads);
  tt_free(twos);
  tt_free(grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_square(tt *input, bool track_grads) {
  tt **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(SQUARE) * sizeof(tt *));
    parents[0] = input;
  }

  tt *t = tt_zeros(input->view->shape, requires_grad);
  t->parents = parents;
  t->op = SQUARE;
  t->_backwards = &_square_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = pow(input->data->buffer[i], 2);
  }
  return t;
}

void _sqrt_backwards(tt *self) {
  tt *copy_input = tt_copy(self->parents[0], false);

  for (int i = 0; i < copy_input->data->size; i++) {
    copy_input->data->buffer[i] = pow(copy_input->data->buffer[i], -.5);
  }
  tt *halfs = tt_fill(self->view->shape, 1.0 / 2, false);
  tt *grads = tt_mul(halfs, copy_input, false);
  tt *mul_self_grads = tt_mul(grads, self->grads, false);
  tt *acc_grads = tt_add(mul_self_grads, self->parents[0]->grads, false);
  tt_free(copy_input);
  tt_free(self->parents[0]->grads);
  tt_free(halfs);
  tt_free(grads);
  tt_free(mul_self_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_sqrt(tt *input, bool track_grads) {
  tt **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(SQRT) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, requires_grad);
  t->parents = parents;
  t->op = SQRT;
  t->_backwards = &_sqrt_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = sqrtf(input->data->buffer[i]);
  }
  return t;
}

void _exp_backwards(tt *self) {
  tt *mul = tt_mul(self, self->grads, false);
  tt *acc_grads = tt_add(mul, self->parents[0]->grads, false);
  tt_free(self->parents[0]->grads);
  tt_free(mul);
  self->parents[0]->grads = acc_grads;
}

tt *tt_exp(tt *input, bool track_grads) {
  tt **parents = NULL;
  bool requires_grad = input->requires_grad && track_grads;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(EXP) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, requires_grad);
  t->parents = parents;
  t->op = EXP;
  t->_backwards = &_exp_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = exp(input->data->buffer[i]);
  }
  return t;
}

void _log_backwards(tt *self) {
  tt *copy_input = tt_zeros(self->parents[0]->view->shape, false);
  for (int i = 0; i < copy_input->data->size; i++) {
    copy_input->data->buffer[i] = 1.0f / self->parents[0]->data->buffer[i];
  }
  tt *mul = tt_mul(copy_input, self->grads, false);
  tt *acc_grads = tt_add(mul, self->parents[0]->grads, false);
  tt_free(self->parents[0]->grads);
  tt_free(mul);
  tt_free(copy_input);
  self->parents[0]->grads = acc_grads;
}

tt *tt_log(tt *input, bool track_grads) {
  tt **parents = NULL;
  bool requires_grad = track_grads && input->requires_grad;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(LOG) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, requires_grad);
  t->parents = parents;
  t->op = LOG;
  t->_backwards = &_log_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = logf(input->data->buffer[i]);
  }
  return t;
}

void _reciprocal_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    // - 1 / x^2
    tt *grads_neg = tt_neg(self->grads, false);
    tt *grads_mul = tt_mul(grads_neg, self, false);
    tt *grads_square = tt_mul(grads_mul, self, false);
    tt *acc_grads_1 = tt_add(grads_square, self->parents[0]->grads, false);
    tt_free(self->parents[0]->grads);
    tt_free(grads_neg);
    tt_free(grads_mul);
    tt_free(grads_square);
    self->parents[0]->grads = acc_grads_1;
  }
}

tt *tt_reciprocal(tt *a, bool track_grads) {
  intarray *copy = intarray_copy(a->view->shape);

  bool requires_grad = track_grads && a->requires_grad;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(tensor_op_operands(RECIPROCAL) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = RECIPROCAL;
  t->_backwards = &_reciprocal_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = 1.0f / a->data->buffer[i];
  }
  return t;
}
