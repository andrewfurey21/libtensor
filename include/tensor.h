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

typedef struct {
  int32_t *items;
  uint32_t size;
} intarray;

intarray *intarray_build(uint32_t size, ...);
intarray *intarray_zeros(uint32_t size);
intarray *intarray_ones(uint32_t size);
intarray *intarray_add(intarray *a, intarray *b);
uint64_t intarray_prod(intarray *s);
intarray *intarray_copy(intarray *other);
bool intarray_equal(intarray *a, intarray *b);
intarray *intarray_div(intarray *a, intarray *b);
void intarray_free(intarray *s);
void intarray_print(intarray *s);
intarray *intarray_add_one(intarray *s);

typedef struct {
  float *buffer;
  uint64_t refcount;
  uint64_t size;
} storage;

typedef struct {
  intarray *shape;
  intarray *strides;
  uint64_t offset;
} view;

typedef struct tensor tensor;
struct tensor {
  storage *data;
  view *dview;

  tensor **parents;
  void (*_backwards)(tensor *);
  tensor_op op;

  bool requires_grad;
  tensor *grads;
};

// TODO: tostring
// (cache in repr, use inside print), view/reshape

tensor *tensor_zeros(intarray *s, bool requires_grad);
tensor *tensor_ones(intarray *s, bool requires_grad);
tensor *tensor_from_buffer(intarray *s, float *buffer, bool requires_grads);
float tensor_getindex(tensor *self, intarray *s);
void tensor_setindex(tensor *self, intarray *s, float num);
tensor *tensor_fill(intarray *s, float fill_value, bool requires_grad);
tensor *tensor_linspace(intarray *s, float min, float max, bool requires_grad);
tensor *tensor_uniform(intarray *s, float min, float max, bool requires_grad);
tensor *tensor_uniformint(intarray *s, float min, float max,
                          bool requires_grad);
//void tensor_copy_buffer(tensor *dest, tensor *src);
tensor *tensor_copy(tensor *original, bool requires_grad);
void tensor_to_zeros(tensor *t);
void tensor_to_n(tensor *t, float n);
void tensor_print(tensor *t, bool show_buffer, bool show_grads);
// tensor *tensor_view(tensor *tensor, view *view);
void tensor_free(tensor *t);
bool tensor_equal(tensor *a, tensor *b, float rtol, float atol);
tensor *tensor_linear_init(intarray *shape, int in_features,
                           bool requires_grad);
tensor *tensor_conv_init(intarray *shape, int in_channels, int kernel_size,
                         bool requires_grad);

// ops
tensor *tensor_add(tensor *a, tensor *b, bool track_grads);
tensor *tensor_sub(tensor *a, tensor *b, bool track_grads);
tensor *tensor_mul(tensor *a, tensor *b, bool track_grads);
tensor *tensor_sum(tensor *a, int axis, bool track_grads);
tensor *tensor_relu(tensor *a, bool track_grads);
tensor *tensor_reshape(tensor *a, intarray *new_shape, bool track_grads);
tensor *tensor_expand(tensor *a, uint64_t axis, uint64_t amount,
                      bool track_grads);
tensor *tensor_neg(tensor *a, bool track_grads);
tensor *tensor_maxpool2d(tensor *input, int kernel_size, bool track_grads);
tensor *tensor_matmul(tensor *input, tensor *other, bool track_grads);
tensor *tensor_conv2d(tensor *input, tensor *kernels, bool track_grads);
tensor *tensor_square(tensor *input, bool track_grads);
tensor *tensor_sqrt(tensor *input, bool track_grads);
tensor *tensor_exp(tensor *input, bool track_grads);
tensor *tensor_log(tensor *input, bool track_grads);
tensor *tensor_reciprocal(tensor *a, bool track_grads);

//ops backward
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
// tensor *variance(tensor *input, int axis, int correction); // FIXME:
tensor *sparse_categorical_cross_entropy(tensor *input, tensor *Y);

// helpers
int randi(int min, int max);
int envvar(const char *name, int default_value);

// computational graph
typedef struct {
  struct tensor **nodes;
  size_t size;
  bool training;
} graph;

graph *graph_build(tensor *x);
void graph_free(graph *net);
void graph_zeroed(graph *net);
void graph_backprop(graph *net);
void graph_print(graph *net, bool no_buffer, bool show_grads);

// nn
typedef struct {
  float learning_rate;
} toptimizer_params;

typedef struct optimizer optimizer;
struct optimizer {
  graph *net;
  toptimizer_params *opt_params;
  void (*step)(optimizer *optim);
};

// optimizer *toptimizer_build(tensor **params, uint64_t size,
//                              toptimizer_params *opt_params,
//                              void (*step)(optimizer *));
// void toptimizer_free(optimizer *opt);

#endif

#ifdef __cplusplus
}
#endif
