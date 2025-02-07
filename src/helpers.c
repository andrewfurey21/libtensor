#include "../include/tensor.h"
#include "assert.h"
#include "math.h"

int randi(int min, int max) {
  int value = roundf((float)rand() / (float)RAND_MAX * (max - min) + min);
  return value;
}

int envvar(const char *name, int default_value) {
  const char *var = getenv(name);
  if (var == NULL) {
    return default_value;
  }
  char *endptr;
  unsigned long result;
  result = strtoul(var, &endptr, 10);
  assert(*endptr == 0 && "Only accepts ints");
  return result;
}
