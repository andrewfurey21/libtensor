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

void storage_free(storage *data) {
  free(data->buffer);
  free(data);
}

float storage_getitem(storage *data, uint64_t index) {
  assert(index >= 0 && index < data->size);
  return data->buffer[index];
}

void storage_setitem(storage *data, uint64_t index, float val) {
  assert(index >= 0 && index < data->size);
  data->buffer[index] = val;
}

storage *storage_copy(storage *data) {
  return storage_from_buffer(data->size, data->buffer);
}

void storage_to_zeros(storage *data) {
  for (int i = 0; i < data->size; i++) {
    storage_setitem(data, i, 0.0f);
  }
}

float storage_getindex(storage *data, view *vw, intarray *index) {
  uint64_t physical_index = view_index(vw, index);
  return storage_getitem(data, physical_index);
}

void storage_setindex(storage *data, view *vw, intarray *index, float num) {
  uint64_t physical_index = view_index(vw, index);
  storage_setitem(data, physical_index, num);
}
