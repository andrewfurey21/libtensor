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
// get rid of storage/view abstraction.
// theres a lot of repetitive stuff, especially in ops. refactor a bit.
// test a graph, where on input needs grads and the other doesn't.
// rename a/s in function parameters to original_tensor, shape, etc.
// maybe rename backwards functions to like tensor_sum_backwards (not in
// header file)
// double check backwards functions are accumulating gradients and not just
// reseting them
// refactor ttmaxpool2d backwards, so that its only the first 5 nested for loop
// not both
// refactor setting up tensors in ops with a function.
// could probably calcualte dNext/dCurrent in forward pass, then mul by incoming
// gradient in backwards.

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
uint64_t tstorage_logical_to_physical(tt *t, ttuple *logical_index) {
  ttuple *t_strides = t->view->strides;
  assert(logical_index->size == t->data->size);
  assert(logical_index->size == t_strides->size);

  uint64_t index = 0;
  for (int i = 0; i < logical_index->size; i++) {
    index += logical_index->items[i] * t_strides->items[i];
  }
  return index + t->view->offset;
}

void tview_free(tview *view) {
  ttuple_free(view->shape);
  ttuple_free(view->strides);
  free(view);
}

tt *tt_zeros(ttuple *s, bool requires_grad) {
  uint64_t size = ttuple_prod(s);
  assert(size != 0);

  ttuple *copy = ttuple_copy(s);
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
  t->view->strides = ttuple_ones(copy->size);
  t->view->offset = 0;

  t->data = data;
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;
  return t;
}

tt *tt_ones(ttuple *s, bool requires_grad) {
  tt *ones = tt_zeros(s, requires_grad);
  for (size_t i = 0; i < ones->data->size; i++) {
    tstorage_setitem(ones->data, i, 1.0f);
  }
  return ones;
}

tt *tt_from_buffer(ttuple *s, float *buffer, bool requires_grad) {
  uint64_t size = ttuple_prod(s);
  tstorage *data = tstorage_from_buffer(size, buffer);

  tt *ret = (tt *)malloc(sizeof(tt));
  ttuple *copy = ttuple_copy(s);
  ttuple *strides = ttuple_ones(copy->size);

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

float tt_getindex(tt *self, ttuple *s) {
  ttuple *self_shape = self->view->shape->size < s->size ? ttuple_add_one(self->view->shape) : self->view->shape;
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  assert(index < ttuple_prod(self_shape));
  return self->data->buffer[index];
}

void tt_setindex(tt *self, ttuple *s, float num) {
  ttuple *self_shape = self->view->shape->size < s->size ? ttuple_add_one(self->view->shape) : self->view->shape;
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  assert(index < ttuple_prod(self->view->shape));
  self->data->buffer[index] = num;
}

tt *tt_fill(ttuple *s, float fill_value, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    tstorage_setitem(t->data, i, fill_value);
  }
  return t;
}

tt *tt_linspace(ttuple *s, float min, float max, bool requires_grad) {
  int steps = ttuple_prod(s);
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = min + i * ((max - min) / (steps - 1));
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniform(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (float)rand() / (float)RAND_MAX * (max - min) + min;
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniformint(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_uniform(s, min, max, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = round(tstorage_getitem(t->data, i));
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_copy(tt *original, bool requires_grad) {
  ttuple *shape = ttuple_copy(original->view->shape);
  ttuple *strides = ttuple_copy(original->view->strides);

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
  ttuple_print(t->view->shape);
  if (t->requires_grad) {
    print_op_string(t->op);
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
    ttuple_print(t->grads->view->shape);
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

void tt_free_parents(tt *t) {
  for (int i = 0; i < top_radix(t->op); i++) {
    tt_free(t->parents[i]);
  }
  free(t->parents);
}

bool tt_equal(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape));
  for (int i = 0; i < ttuple_prod(a->view->shape); i++) {
    if (fabs(a->data->buffer[i] - b->data->buffer[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

void tt_destroy_grads(tt *t) {
  t->requires_grad = false;
  tt_free(t->grads);
  t->_backwards = NULL;
  t->op = NOOP;
  tt_free_parents(t);
}

tt* tt_linear_init(ttuple* shape, int in_features, bool requires_grad) {
  float bound = 1.0f / sqrtf((float)in_features);
  return tt_uniform(shape, -bound, bound, requires_grad);
}

tt* tt_conv_init(ttuple* shape, int in_channels, int kernel_size, bool requires_grad) {
  float scale = 1.0f / sqrtf((float)(in_channels * kernel_size));
  return tt_uniform(shape, -scale, scale, requires_grad);
}

void _add_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads);
    // tt_destroy_grads(grads_0); // TODO: check if grads before doing
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_add(self->grads, self->parents[1]->grads);
    // tt_destroy_grads(grads_1);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}
// TODO: need to add track_gradients
// tt_add(tt* a, tt* b, bool track_gradients);
tt *tt_add(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->view->shape); // TODO: free copy!!
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(ADD) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  ttuple_free(copy);
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
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads);
    // tt_destroy_grads(grads_0);
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_sub(self->parents[1]->grads, self->grads);
    // tt_destroy_grads(grads_1);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}

tt *tt_sub(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->view->shape); // TODO: free copy??
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(SUB) * sizeof(tt *));
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
    tt *grads_0 = tt_mul(self->grads, self->parents[1]);
    tt *acc_grads_0 = tt_add(grads_0, self->parents[0]->grads);
    // tt_destroy_grads(acc_grads_0); // this solution doesn't work, but need to
    // fix grads having grads
    tt_free(self->parents[0]->grads);
    tt_free(grads_0);
    self->parents[0]->grads = acc_grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_mul(self->grads, self->parents[0]);
    tt *acc_grads_1 = tt_add(grads_1, self->parents[1]->grads);
    // tt_destroy_grads(acc_grads_1);
    tt_free(self->parents[1]->grads);
    tt_free(grads_1);
    self->parents[1]->grads = acc_grads_1;
  }
}

tt *tt_mul(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->view->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(MUL) * sizeof(tt *));
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
  ttuple *unit_shape = ttuple_build(1, 1);

  ttuple *self_shape = self->view->shape;
  ttuple *parent_shape = self->parents[0]->view->shape;
  if (ttuple_equal(unit_shape, self_shape)) {
    tt *expanded_grads =
        tt_fill(parent_shape, self->grads->data->buffer[0], false);
    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads);
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

    ttuple *current = ttuple_zeros(parent_shape->size);
    uint64_t along_axis = parent_shape->items[expand_axis];
    for (uint64_t i = 0; i < self->grads->data->size; i++) {
      // expanding
      for (uint64_t j = 0; j < along_axis; j++) {
        ttuple *current_grads = ttuple_copy(current);
        current_grads->items[expand_axis] = 0;
        float num = tt_getindex(self->grads, current_grads);
        tt_setindex(expanded_grads, current, num);
        current->items[expand_axis]++;
        ttuple_free(current_grads);
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

    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads);
    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  }
  ttuple_free(unit_shape);
}

// axis=-1 => sum up all elements
// currently always keepdims, except for axis=-1
// could seriously use some tests here
tt *tt_sum(tt *a, int axis) {
  assert(axis >= -1 && axis < (int)a->view->shape->size);
  ttuple *new_shape;
  if (axis == -1) {
    new_shape = ttuple_build(1, 1);
  } else {
    new_shape = ttuple_copy(a->view->shape);
    new_shape->items[axis] = 1;
  }

  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(SUM_REDUCE) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(new_shape, a->requires_grad);
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
    ttuple *stride = ttuple_zeros(a->view->shape->size);
    stride->items[axis] = 1;

    uint64_t along_axis = a->view->shape->items[axis];
    uint64_t num_accumulate = ttuple_prod(a->view->shape) / along_axis;
    ttuple *current = ttuple_zeros(a->view->shape->size);
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
  tt *mul_grads = tt_mul(self->grads, grads);
  tt *acc_grads = tt_add(self->parents[0]->grads, mul_grads);
  tt_free(grads);
  tt_free(self->parents[0]->grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_relu(tt *a) {
  ttuple *copy = ttuple_copy(a->view->shape);
  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RELU) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, a->requires_grad);
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
  tt *grads = tt_reshape(self->grads, self->parents[0]->view->shape);
  tt *acc_grads = tt_add(grads, self->parents[0]->grads);
  free(grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_reshape(tt *a, ttuple *new_shape) {
  ttuple *new_shape_copy = ttuple_copy(new_shape);
  assert(ttuple_prod(new_shape) == ttuple_prod(a->view->shape));
  tt **parents = NULL;
  tt *reshaped_grads = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RESHAPE) * sizeof(tt *));
    parents[0] = a;
    // TODO: shouldn't op be RESHAPE of grad then? why isn't it? it shouldn't be
    // anyways.
    reshaped_grads = tt_reshape(a->grads, new_shape_copy);
  }
  tt *t = tt_copy(a, a->requires_grad);
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
  ttuple *div = ttuple_div(self->view->shape, self->parents[0]->view->shape);
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
  uint64_t num_accumulate = ttuple_prod(self->view->shape) / along_axis;
  ttuple *current = ttuple_zeros(self->view->shape->size);

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
tt *tt_expand(tt *original_tensor, uint64_t axis, uint64_t factor) {
  ttuple *new_shape = ttuple_copy(original_tensor->view->shape);
  assert(axis >= 0 && axis < new_shape->size &&
         "Axis to expand is out of range.");
  assert(factor > 0 && "Expanding factor must be greater than 0");
  assert(new_shape->items[axis] == 1 && "Cannot expand [axis]!=1");

  // calculate new shape here
  new_shape->items[axis] *= factor; // TODO: check overflows.

  tt **parents = NULL;
  if (original_tensor->requires_grad) {
    parents = (tt **)malloc(top_radix(EXPAND) * sizeof(tt *));
    parents[0] = original_tensor;
  }

  tt *expanded_tensor = tt_zeros(new_shape, original_tensor->requires_grad);
  expanded_tensor->parents = parents;
  expanded_tensor->op = EXPAND;
  expanded_tensor->_backwards = &_expand_backwards;

  // expand here
  ttuple *expanded_index = ttuple_zeros(expanded_tensor->view->shape->size);
  uint64_t along_axis = new_shape->items[axis];

  for (uint64_t i = 0; i < expanded_tensor->data->size; i++) {
    // expanding (like _sum_backwards)
    for (uint64_t j = 0; j < along_axis; j++) {
      ttuple *original_index = ttuple_copy(expanded_index);
      original_index->items[axis] = 0;
      float num = tt_getindex(original_tensor, original_index);
      tt_setindex(expanded_tensor, expanded_index, num);
      expanded_index->items[axis]++;
      ttuple_free(original_index);
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
  tt *mul_grads = tt_mul(grads, self->grads);
  tt *acc_grads = tt_add(mul_grads, self->parents[0]->grads);
  tt_free(self->parents[0]->grads);
  tt_free(grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_neg(tt *a) {
  ttuple *shape = ttuple_copy(a->view->shape);
  tt *t = tt_zeros(shape, a->requires_grad);

  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(NEG) * sizeof(tt *));
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

  ttuple *self_shape = self->view->shape;
  ttuple *parent_shape = self->parents[0]->view->shape;
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
  ttuple *index = ttuple_zeros(parent_shape->size);
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
  ttuple_free(index);

  index = ttuple_zeros(parent_shape->size);
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
          ttuple *max_index = ttuple_copy(index);
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
              ttuple_free(max_index);
              max_index = ttuple_copy(index);
            } else {
              tt_setindex(pooled_grads, index, 0);
            }
          }
          ttuple_free(max_index);
        }
      }
    }
  }
  ttuple_free(index);

  tt *expanded_by_pooled_grads = tt_mul(expanded_self_grad, pooled_grads);
  tt *accumulated_grads =
      tt_add(self->parents[0]->grads, expanded_by_pooled_grads);
  tt_free(expanded_self_grad);
  tt_free(pooled_grads);
  self->parents[0]->grads = accumulated_grads;
}

// NOTE:
// assuming input is divisible by kernel size
// stride is kernel size
// no dilation, padding. ceilmode=False.
// 4d, 3d, 2d only.
tt *tt_maxpool2d(tt *input, int kernel_size) {
  ttuple *input_shape = input->view->shape;
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

  ttuple *new_shape = ttuple_copy(input_shape);
  new_shape->items[end_index] = new_width;
  new_shape->items[end_index - 1] = new_height;

  tt **parents = NULL;
  if (input->requires_grad) {
    parents = (tt **)malloc(top_radix(MAX_POOL_2D) * sizeof(tt *));
    parents[0] = input;
  }
  tt *output = tt_zeros(new_shape, input->requires_grad);
  output->parents = parents;
  output->op = MAX_POOL_2D;
  output->_backwards = &_maxpool2d_backwards;

  int dims = input_shape->size;
  int channels = input_shape->size > 2 ? input_shape->items[dims - 3] : 0;
  int batches = input_shape->size > 3 ? input_shape->items[dims - 4] : 0;

  // NOTE: this looks really bad.
  ttuple *index = ttuple_copy(input_shape);
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
  ttuple_free(index);
  return output;
}

void _matmul_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
  }
  if (self->parents[1]->requires_grad) {
  }
}

// for now, a can have 2d
// b can have 2d or 3d (for batches)
// need to do broadcasting
tt *tt_matmul(tt *a, tt *b) {
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

  ttuple *new_shape;
  if (b_size == 3) {
    new_shape = ttuple_build(3, bs, ah, bw);
  } else {
    new_shape = ttuple_build(2, ah, bw);
  }

  bool requires_grad = a->requires_grad || b->requires_grad;
  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(MATMUL) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(new_shape, requires_grad);
  ttuple_free(new_shape);
  t->parents = parents;
  t->op = MATMUL;
  t->_backwards = &_matmul_backwards;

  ttuple *ai = ttuple_zeros(2);
  ttuple *bi = ttuple_zeros(3);
  ttuple *oi = ttuple_zeros(3);

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
  ttuple_free(ai);
  ttuple_free(bi);
  ttuple_free(oi);
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
    ttuple *input_grad_index = ttuple_zeros(4);
    ttuple *kernel_index = ttuple_zeros(4);
    ttuple *output_grad_index = ttuple_zeros(4);
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
    tt *acc_grads = tt_add(grads, self->parents[0]->grads);

    tt_free(grads);
    tt_free(self->parents[0]->grads);

    ttuple_free(kernel_index);
    ttuple_free(input_grad_index);
    ttuple_free(output_grad_index);

    self->parents[0]->grads = acc_grads;
  }

  // kernel gradients
  if (self->parents[1]->requires_grad) {
    tt *grads = tt_zeros(self->parents[1]->view->shape, false);
    ttuple *input_index = ttuple_zeros(4);
    ttuple *kernel_grad_index = ttuple_zeros(4);
    ttuple *output_grad_index = ttuple_zeros(4);
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
    tt *acc_grads = tt_add(grads, self->parents[1]->grads);

    tt_free(grads);
    tt_free(self->parents[1]->grads);

    ttuple_free(kernel_grad_index);
    ttuple_free(input_index);
    ttuple_free(output_grad_index);

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
tt *tt_conv2d(tt *input, tt *kernels) {
  ttuple *input_shape = input->view->shape;
  assert(input_shape->size == 4);

  ttuple *kernels_shape = kernels->view->shape;
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

  ttuple *out_shape = ttuple_build(4, batch_size, cout, hout, wout);

  bool requires_grad = input->requires_grad || kernels->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(CONV_2D) * sizeof(tt *));
    parents[0] = input;
    parents[1] = kernels;
  }

  tt *output = tt_zeros(out_shape, requires_grad);
  output->parents = parents;
  output->op = CONV_2D;
  output->_backwards = &_conv2d_backwards;

  ttuple *input_index = ttuple_zeros(4);
  ttuple *kernel_index = ttuple_zeros(4);
  ttuple *output_index = ttuple_zeros(4);
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
  ttuple_free(input_index);
  ttuple_free(output_index);
  ttuple_free(kernel_index);
  return output;
}

void _square_backwards(tt *self) {
  tt *twos = tt_fill(self->view->shape, 2, false);
  tt *grads =
      tt_mul(twos, self->parents[0]); // TODO: parents might have grads, so need
                                      // to clear grads->grads later
  tt *mul_self_grads = tt_mul(grads, self->grads);
  tt *acc_grads = tt_add(mul_self_grads, self->parents[0]->grads);
  tt_free(self->parents[0]->grads);
  tt_free(twos);
  tt_free(grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_square(tt *input) {
  tt **parents = NULL;
  if (input->requires_grad) {
    parents = (tt **)malloc(top_radix(SQUARE) * sizeof(tt *));
    parents[0] = input;
  }

  tt *t = tt_zeros(input->view->shape, input->requires_grad);
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
  tt *grads = tt_mul(halfs, copy_input);
  tt *mul_self_grads = tt_mul(grads, self->grads);
  tt *acc_grads = tt_add(mul_self_grads, self->parents[0]->grads);
  tt_free(copy_input);
  tt_free(self->parents[0]->grads);
  tt_free(halfs);
  tt_free(grads);
  tt_free(mul_self_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_sqrt(tt *input) {
  tt **parents = NULL;
  if (input->requires_grad) {
    parents = (tt **)malloc(top_radix(SQRT) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, input->requires_grad);
  t->parents = parents;
  t->op = SQRT;
  t->_backwards = &_sqrt_backwards;
  for (uint64_t i = 0; i < input->data->size; i++) {
    t->data->buffer[i] = sqrtf(input->data->buffer[i]);
  }
  return t;
}

void _exp_backwards(tt *self) {
  tt *mul = tt_mul(self, self->grads); // TODO: this grad might have grads now
  tt *acc_grads = tt_add(mul, self->parents[0]->grads);
  tt_free(self->parents[0]->grads);
  tt_free(mul);
  self->parents[0]->grads = acc_grads;
}

tt *tt_exp(tt *input) {
  tt **parents = NULL;
  if (input->requires_grad) {
    parents = (tt **)malloc(top_radix(EXP) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, input->requires_grad);
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
  tt *mul =
      tt_mul(copy_input, self->grads); // TODO: this grad might have grads now
  tt *acc_grads = tt_add(mul, self->parents[0]->grads);
  tt_free(self->parents[0]->grads);
  tt_free(mul);
  tt_free(copy_input);
  self->parents[0]->grads = acc_grads;
}

tt *tt_log(tt *input) {
  tt **parents = NULL;
  if (input->requires_grad) {
    parents = (tt **)malloc(top_radix(LOG) * sizeof(tt *));
    parents[0] = input;
  }
  tt *t = tt_zeros(input->view->shape, input->requires_grad);
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
    tt *grads_neg = tt_neg(self->grads);
    tt *grads_mul = tt_mul(grads_neg, self);
    tt *grads_square = tt_mul(grads_mul, self);
    tt *acc_grads_1 = tt_add(grads_square, self->parents[0]->grads);
    tt_free(self->parents[0]->grads);
    tt_free(grads_neg);
    tt_free(grads_mul);
    tt_free(grads_square);
    self->parents[0]->grads = acc_grads_1;
  }
}

tt *tt_reciprocal(tt *a) {
  ttuple *copy = ttuple_copy(a->view->shape);

  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RECIPROCAL) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, a->requires_grad);
  t->parents = parents;
  t->op = RECIPROCAL;
  t->_backwards = &_reciprocal_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = 1.0f / a->data->buffer[i];
  }
  return t;
}
