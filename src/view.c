#include "../include/tensor.h"
#include "assert.h"

view *view_new(intarray *shape) {
  intarray *copy = intarray_copy(shape);
  view *v = (view *)malloc(sizeof(view));
  v->shape = copy;
  return v;
}

uint64_t view_index(view *vw, intarray *index) {
  assert(index->size >= vw->shape->size);
  uint64_t physical_index = 0;
  bool pad_left = vw->shape->size < index->size;
  intarray *padded_shape =
      pad_left ? intarray_pad_left(vw->shape, index->size) : vw->shape;

  // indexing here
  for (int i = 0; i < index->size; i++) {
    uint64_t mul = 1;
    for (int j = i + 1; j < index->size; j++) {
      mul *= padded_shape->items[j];
    }
    physical_index += mul * index->items[i];
  }

  assert(physical_index < intarray_prod(vw->shape));
  if (pad_left) {
    intarray_free(padded_shape);
  }
  return physical_index;
}

view* view_copy(view *vw) {
  return view_new(vw->shape);
}

bool view_equal(view *a, view *b) {
  return intarray_equal(a->shape, b->shape);
}

void view_free(view *vw) {
  intarray_free(vw->shape);
  free(vw);
}
