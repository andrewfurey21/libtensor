#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"

view *view_new(intarray *shape) {
  intarray *copy = intarray_copy(shape);
  view *v = (view *)malloc(sizeof(view));
  v->shape = copy;
  return v;
}

uint64_t view_index(view *v, intarray *index) {
  assert(index->size >= v->shape->size);
  uint64_t physical_index = 0;
  bool pad_left = v->shape->size < index->size;
  intarray *padded_shape =
      pad_left ? intarray_pad_left(v->shape, index->size) : v->shape;

  // indexing here
  for (int i = 0; i < index->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < index->size; j++) {
      mul *= padded_shape->items[j];
    }
    physical_index += mul * index->items[i];
  }

  assert(physical_index < intarray_prod(v->shape));
  if (pad_left) {
    intarray_free(padded_shape);
  }
  return physical_index;
}

view* view_copy(view *v) {
  return view_new(v->shape);
}

void view_free(view *view) {
  intarray_free(view->shape);
  free(view);
}
