#include "../include/tensor.h"
#include "assert.h"

view* view_new(intarray* shape) {
  intarray* copy = intarray_copy(shape);
  view *v = (view *)malloc(sizeof(view));
  v->shape = copy;
  return v;
}

void view_free(view *view) {
  intarray_free(view->shape);
  free(view);
}
