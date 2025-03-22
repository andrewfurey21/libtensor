#include "../include/tensor.h"
#include "assert.h"
#include "string.h"

storage *storage_zeros(uint64_t buffer_length) {
  float *buffer = (float *)calloc(buffer_length, sizeof(float));
  storage *s = (storage *)malloc(sizeof(storage));
  s->size = buffer_length;
  s->buffer = buffer;
  return s;
}

storage *storage_from_buffer(uint64_t size, float *buffer) {
  float *buffer_copy = (float *)malloc(sizeof(float) * size);
  memcpy(buffer_copy, buffer, sizeof(float) * size);
  storage *data = (storage *)malloc(sizeof(storage));
  data->size = size;
  data->buffer = buffer_copy;
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

storage *storage_copy(storage *s) {
  return storage_from_buffer(s->size, s->buffer);
}

void storage_to_zeros(storage *s) {
  free(s->buffer);
  s->buffer = (float *)calloc(s->size, sizeof(float));
}

float storage_getindex(storage *data, view* v, intarray *index) {
  uint64_t physical_index = view_index(v, index);
  return storage_getitem(data, physical_index);
}

void storage_setindex(storage *data, view *v, intarray* index, float num) {
  uint64_t physical_index = view_index(v, index);
  storage_setitem(data, physical_index, num);
}
