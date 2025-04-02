#ifdef __cplusplus
extern "C" {
#endif

#include "stdarg.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdlib.h"

#ifndef _TENSOR_H
#define _TENSOR_H

#define MAX_ITEMS 4

typedef enum {
  NOOP = 0,
  RELU,
  SUM_REDUCE,
  RESHAPE,
  EXPAND,
  ADD,
  SUB,
  MUL,
  RECIPROCAL,
  MAX_POOL_2D,
  NEG,
  CONV_2D,
  MATMUL,
  SQUARE,
  SQRT,
  EXP,
  LOG,
} tensor_op;

size_t tensor_op_operands(tensor_op op);
void print_tensor_op(tensor_op op);

// intarray

typedef struct {
  int32_t *items;
  uint32_t size;
} intarray;

intarray *intarray_build(uint32_t size, ...);
intarray *intarray_zeros(uint32_t size);
intarray *intarray_ones(uint32_t size);
intarray *intarray_add(intarray *a, intarray *b);
uint64_t intarray_prod(intarray *a);
intarray *intarray_copy(intarray *a);
bool intarray_equal(intarray *a, intarray *b);
intarray *intarray_div(intarray *a, intarray *b);
void intarray_free(intarray *a);
void intarray_print(intarray *a);
intarray *intarray_pad_left(intarray *a, int new_size);
intarray *intarray_squeeze(intarray *a);

// View

typedef struct {
  intarray *shape;
} view;

view *view_new(intarray *shape);
uint64_t view_index(view *vw, intarray *index);
view *view_copy(view *vw);
bool view_equal(view *a, view *b);
void view_free(view *vw);

// Storage

typedef struct {
  float *buffer;
  uint64_t size;
} storage;

storage *storage_zeros(uint64_t buffer_length);
storage *storage_from_buffer(uint64_t size, float *buffer);
void storage_free(storage *data);
float storage_getitem(storage *data, uint64_t index);
void storage_setitem(storage *data, uint64_t index, float val);
storage *storage_copy(storage *data);
void storage_to_zeros(storage *data);
float storage_getindex(storage *data, view *vw, intarray *index);
void storage_setindex(storage *data, view *vw, intarray *index, float num);

// tensor

typedef struct tensor tensor;
struct tensor {
  storage *data;
  view *vw;

  tensor **parents;
  void (*_backwards)(tensor *);
  tensor_op op;

  bool requires_grad;
  tensor *grads;
};

// TODO: tostring
// (cache in repr, use inside print), view/reshape
tensor *tensor_zeros(intarray *shape, bool requires_grad);
tensor *tensor_ones(intarray *shape, bool requires_grad);
tensor *tensor_from_buffer(intarray *shape, float *buffer, bool requires_grads);
float tensor_getindex(tensor *input, intarray *index);
void tensor_setindex(tensor *input, intarray *index, float num);
tensor *tensor_fill(intarray *shape, float fill_value, bool requires_grad);
tensor *tensor_linspace(intarray *shape, float min, float max,
                        bool requires_grad);
tensor *tensor_uniform(intarray *shape, float min, float max,
                       bool requires_grad);
tensor *tensor_uniformint(intarray *shape, float min, float max,
                          bool requires_grad);
tensor *tensor_copy(tensor *original, bool requires_grad);
void tensor_to_zeros(tensor *input);
void tensor_to_n(tensor *input, float n);
void tensor_copy_buffer(tensor *dest, tensor *src);
void tensor_print(tensor *input, bool show_buffer, bool show_grads);
void tensor_free(tensor *input);
bool tensor_equal(tensor *a, tensor *b, float rtol, float atol);
tensor *tensor_linear_init(intarray *shape, int in_features,
                           bool requires_grad);
tensor *tensor_conv_init(intarray *shape, int in_channels, int kernel_size,
                         bool requires_grad);

// ops
tensor *tensor_add(tensor *a, tensor *b, bool track_grads);
tensor *tensor_sub(tensor *a, tensor *b, bool track_grads);
tensor *tensor_mul(tensor *a, tensor *b, bool track_grads);
tensor *tensor_sum(tensor *input, int axis, bool track_grads);
tensor *tensor_relu(tensor *input, bool track_grads);
tensor *tensor_reshape(tensor *input, intarray *new_shape, bool track_grads);
tensor *tensor_expand(tensor *input, uint64_t axis, uint64_t amount,
                      bool track_grads);
tensor *tensor_neg(tensor *input, bool track_grads);
tensor *tensor_maxpool2d(tensor *input, int kernel_size, bool track_grads);
tensor *tensor_matmul(tensor *a, tensor *b, bool track_grads);
tensor *tensor_conv2d(tensor *input, tensor *kernels, bool track_grads);
tensor *tensor_square(tensor *input, bool track_grads);
tensor *tensor_sqrt(tensor *input, bool track_grads);
tensor *tensor_exp(tensor *input, bool track_grads);
tensor *tensor_log(tensor *input, bool track_grads);
tensor *tensor_reciprocal(tensor *input, bool track_grads);

// ops backward
void _add_backwards(tensor *self);
void _sub_backwards(tensor *self);
void _mul_backwards(tensor *self);
void _sum_backwards(tensor *self);
void _relu_backwards(tensor *self);
void _reshape_backwards(tensor *self);
void _expand_backwards(tensor *self);
void _neg_backwards(tensor *self);
void _maxpool2d_backwards(tensor *self);
void _matmul_backwards(tensor *self);
void _conv2d_backwards(tensor *self);
void _square_backwards(tensor *self);
void _sqrt_backwards(tensor *self);
void _exp_backwards(tensor *self);
void _log_backwards(tensor *self);
void _reciprocal_backwards(tensor *self);

// functions
tensor *flatten(tensor *input, int start_dim);
tensor *mean(tensor *input, int axis);
tensor *variance(tensor *input, int axis, int correction);
tensor *one_hot_encode(tensor *batch, int size);
tensor *cross_entropy(tensor *input, tensor *Y);
tensor *log_softmax(tensor *input);

// computational graph
typedef struct {
  struct tensor **nodes;
  size_t size;
} graph;

graph *graph_build(tensor *x);
void graph_free(graph *net);
void graph_zeroed(graph *net);
void graph_backprop(graph *net);
void graph_print(graph *net, bool no_buffer, bool show_grads);

// nn
void sgd_step(graph *network, float learning_rate);

// helpers
int randi(int min, int max);
int envvar(const char *name, int default_value);

#endif
#ifdef __cplusplus
}
#endif
